"""Bounded website/sitemap discovery with robots rules and conditional HTML fetches."""

import gzip
import io
import ipaddress
import os
import socket
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlsplit
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
from defusedxml import ElementTree

from .text import canonical_url, normalize, public_page

AGENT = "StartupFloraRAG/2.0"
MAX_BYTES = 5_000_000


@dataclass
class Fetch:
    """Carry a bounded HTTP response without retaining an open network connection."""

    url: str
    status: int
    body: bytes
    headers: dict


@dataclass
class Page:
    """Represent parsed public HTML or an unchanged conditional response."""

    url: str
    status: int
    title: str = ""
    text: str = ""
    links: list[str] = field(default_factory=list)
    headers: dict = field(default_factory=dict)


class Website:
    """Fetch only administrator-approved public HTTPS hosts; visitors cannot call this crawler."""

    def __init__(self, settings, client=None):
        """Create a reusable HTTP connection pool and per-host robots/pacing state."""
        self.s = settings
        self.client = client or httpx.Client(
            timeout=httpx.Timeout(15, connect=8),
            follow_redirects=False,
            headers={"User-Agent": AGENT},
            trust_env=False,
        )
        self.policies, self.last_request = {}, {}
        self.discovery_warnings = []
        self.discovery_limited = False
        # Opt-in debug output: set CRAWL_DEBUG_HTML=1 in PowerShell before running.
        # In debug mode, request a full page instead of a body-less HTTP 304.
        self.debug_html = os.environ.get("CRAWL_DEBUG_HTML", "").lower() in (
            "1",
            "true",
            "yes",
            "on",
        )

    def _debug(self, message):
        """Print crawler diagnostics to stderr, safely handling Windows console encoding."""
        if self.debug_html:
            encoding = getattr(sys.stderr, "encoding", None) or "utf-8"
            safe = (
                str(message)
                .encode(encoding, errors="backslashreplace")
                .decode(encoding)
            )
            print(safe, file=sys.stderr, flush=True)

    def checked_url(self, url):
        """Reject non-allowlisted hosts and private/reserved DNS targets on each redirect."""
        safe = canonical_url(url, self.s.crawl_hosts)
        addresses = socket.getaddrinfo(
            urlsplit(safe).hostname, 443, type=socket.SOCK_STREAM
        )
        if not addresses or any(
            not ipaddress.ip_address(a[4][0]).is_global for a in addresses
        ):
            raise ValueError("Refusing a non-public website address")
        return safe

    def _wait(self, host):
        """Rate-limit each host and honor supported robots Crawl-delay values."""
        delay = self.s.crawl_delay
        policy = self.policies.get(host)
        if policy:
            configured = policy.crawl_delay(AGENT)
            if configured:
                if configured > 60:
                    raise ValueError(
                        "robots Crawl-delay exceeds this crawler's bounded wait; use a slower admin job"
                    )
                delay = max(delay, float(configured))
        remaining = self.last_request.get(host, 0) + delay - time.monotonic()
        if remaining > 0:
            time.sleep(remaining)
        self.last_request[host] = time.monotonic()

    def fetch(self, url, headers=None, is_robots=False):
        """Bound downloads, redirects and transient retries; verify robots before content requests."""
        for _ in range(5):
            url = self.checked_url(url)
            if not is_robots and not self.permitted(url):
                raise PermissionError("robots.txt disallows this URL")
            host = urlsplit(url).hostname
            response_data = None
            for attempt in range(3):
                self._wait(host)
                try:
                    with self.client.stream(
                        "GET", url, headers=headers or {}
                    ) as response:
                        data = bytearray()
                        for block in response.iter_bytes():
                            data.extend(block)
                            if len(data) > MAX_BYTES:
                                raise ValueError(
                                    "Download exceeds the configured size limit"
                                )
                        response_data = Fetch(
                            str(response.url),
                            response.status_code,
                            bytes(data),
                            dict(response.headers),
                        )
                except (httpx.TimeoutException, httpx.NetworkError):
                    if attempt == 2:
                        raise
                    time.sleep(0.25 * (attempt + 1))
                    continue
                if (
                    response_data.status not in (429, 500, 502, 503, 504)
                    or attempt == 2
                ):
                    break
                time.sleep(0.25 * (attempt + 1))
            if response_data.status in (301, 302, 303, 307, 308):
                location = response_data.headers.get("location")
                if not location:
                    raise ValueError("Redirect has no destination")
                url = urljoin(url, location)
                # Do not forward validators from one resource to another.
                headers = None
                continue
            self._debug(
                f"\n[FETCH] {response_data.url}\n"
                f"[STATUS] {response_data.status}\n"
                f"[TYPE] {response_data.headers.get('content-type', 'unknown')}\n"
                f"[BYTES] {len(response_data.body)}"
            )
            return response_data
        raise ValueError("Too many redirects")

    def policy(self, url):
        """Cache robots policy; permit missing robots files but fail closed on other fetch failures."""
        host = urlsplit(url).hostname
        if host not in self.policies:
            robot = RobotFileParser("https://" + host + "/robots.txt")
            response = self.fetch(robot.url, is_robots=True)
            if response.status in (401, 403):
                robot.disallow_all = True
            elif response.status == 404:
                robot.allow_all = True
            elif 200 <= response.status < 300:
                robot.parse(
                    response.body.decode("utf-8", errors="replace").splitlines()
                )
            else:
                raise ValueError("Cannot verify robots.txt; refusing to crawl")
            self.policies[host] = robot
        return self.policies[host]

    def permitted(self, url):
        """Apply the correct per-host robots policy to a public URL."""
        return self.policy(url).can_fetch(AGENT, url)

    def sitemap_urls(self, start, explicit=None):
        """Discover nested XML/gzip sitemap URLs with deduplication and hard resource limits."""
        root = "https://" + urlsplit(start).hostname
        hints = self.policy(start).site_maps() or []
        maps = deque(
            [explicit]
            if explicit
            else hints + [root + "/sitemap.xml", root + "/sitemap_index.xml"]
        )
        seen_maps, found = set(), {}
        while maps and len(seen_maps) < self.s.max_sitemaps:
            candidate = maps.popleft()
            try:
                candidate = canonical_url(candidate, self.s.crawl_hosts)
                if candidate in seen_maps:
                    continue
                seen_maps.add(candidate)
                result = self.fetch(candidate)
                if result.status != 200:
                    raise ValueError("Sitemap HTTP failure")
                data = result.body
                if data.startswith(b"\x1f\x8b"):
                    with gzip.GzipFile(fileobj=io.BytesIO(data)) as stream:
                        data = stream.read(MAX_BYTES + 1)
                    if len(data) > MAX_BYTES:
                        raise ValueError("Expanded sitemap is too large")
                if not data.lstrip().startswith(b"<"):
                    # Support simple one-URL-per-line sitemaps too.
                    entries = data.decode("utf-8").splitlines()
                    kind = "urlset"
                else:
                    tree = ElementTree.fromstring(data)
                    kind = tree.tag.rsplit("}", 1)[-1]
                    if kind not in ("urlset", "sitemapindex"):
                        raise ValueError("Unsupported sitemap root")
                    # Direct child loc only: image/news extension locs are not pages.
                    entries = []
                    for entry in tree:
                        for child in entry:
                            if child.tag.rsplit("}", 1)[-1] == "loc" and child.text:
                                entries.append(child.text.strip())
                                break
                for entry in entries:
                    try:
                        url = canonical_url(entry, self.s.crawl_hosts)
                    except ValueError:
                        continue
                    if kind == "sitemapindex":
                        if url not in seen_maps:
                            if len(maps) < self.s.max_sitemaps * 4:
                                maps.append(url)
                            else:
                                self.discovery_limited = True
                    elif public_page(url):
                        if len(found) >= self.s.max_crawl_urls:
                            self.discovery_limited = True
                            break
                        found[url] = None
            except Exception as exc:
                self.discovery_warnings.append(
                    {"url": candidate, "error_type": type(exc).__name__}
                )
        if maps:
            self.discovery_limited = True
        return list(found)

    def page(self, url, previous=None):
        """Use ETag/Last-Modified validators and parse meaningful HTML plus internal links."""
        headers = {}
        if previous and not self.debug_html:
            if previous.get("etag"):
                headers["If-None-Match"] = previous["etag"]
            if previous.get("last_modified"):
                headers["If-Modified-Since"] = previous["last_modified"]
        response = self.fetch(url, headers=headers)
        if response.status in (304, 404, 410):
            self._debug(f"[SKIP HTML] HTTP {response.status}: {response.url}")
            return Page(
                url=response.url,
                status=response.status,
                headers=response.headers,
                links=(previous or {}).get("links", []),
            )
        if response.status != 200:
            raise ValueError(f"Page HTTP status {response.status}")
        if "text/html" not in response.headers.get("content-type", ""):
            raise ValueError("Page is not HTML; import approved PDFs separately")
        soup = BeautifulSoup(response.body, "html.parser")
        # Print before removing tags so scripts/navigation remain visible here.
        # BeautifulSoup formats the downloaded HTML; it does not execute JavaScript.
        if self.debug_html:
            self._debug(f"\n[HTML START] {response.url}\n{soup.prettify()}\n[HTML END]")
        title = soup.title.get_text(" ", strip=True) if soup.title else response.url
        links = {}
        for link in soup.select("a[href]"):
            raw = urljoin(response.url, link["href"])
            try:
                if urlsplit(raw).query:
                    continue
                safe = canonical_url(raw, self.s.crawl_hosts)
                if public_page(safe):
                    links[safe] = None
            except ValueError:
                continue
        for tag in soup.select("script,style,nav,footer,header,noscript,form,svg"):
            tag.decompose()
        body = (
            soup.select_one("main") or soup.select_one("article") or soup.body or soup
        )
        text = normalize(body.get_text("\n", strip=True))
        self._debug(
            f"\n[EXTRACTED TEXT START] {response.url}\n{text}\n[EXTRACTED TEXT END]"
        )
        self._debug(f"[INTERNAL LINKS] {len(links)}\n[TEXT CHARACTERS] {len(text)}")
        return Page(
            url=response.url,
            status=200,
            title=title,
            text=text,
            links=list(links),
            headers=response.headers,
        )

    def close(self):
        """Release pooled HTTP connections after the indexing job completes."""
        self.client.close()
