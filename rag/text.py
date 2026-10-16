"""Deterministic text, URL and ranking helpers independent of a model provider."""

import hashlib
import logging
import re
from collections import defaultdict
from functools import lru_cache
from urllib.parse import unquote, urlsplit, urlunsplit

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

log = logging.getLogger(__name__)


def normalize(text: str) -> str:
    """Remove noisy spacing while preserving paragraph and heading boundaries."""
    text = "\n".join(re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines())
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def digest(text: str) -> str:
    """Create stable cache keys and content fingerprints without retaining raw queries as keys."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@lru_cache(maxsize=1)
def encoding():
    """Load one tokenizer; use a conservative byte count if its first-run download is unavailable."""
    try:
        return tiktoken.get_encoding("cl100k_base")
    except Exception:
        log.warning("Tokenizer unavailable; using conservative UTF-8 byte budgets")
        return None


def token_count(text: str) -> int:
    """Measure cl100k tokens, or a conservative UTF-8 byte bound when offline."""
    codec = encoding()
    return len(codec.encode(text, disallowed_special=())) if codec else len(text.encode("utf-8"))


def canonical_url(url: str, hosts: tuple[str, ...]) -> str:
    """Allow only configured public HTTPS hosts; drop tracking/query/fragment variants."""
    p = urlsplit(url.strip())
    if (
        len(url) > 3000
        or p.scheme != "https"
        or p.hostname not in hosts
        or p.username
        or p.password
    ):
        raise ValueError("Only configured HTTPS website hosts are allowed")
    if p.port not in (None, 443) or "\\" in url or any(ord(c) < 32 for c in url):
        raise ValueError("Invalid website URL")
    return urlunsplit(("https", p.hostname, p.path or "/", "", ""))


def public_page(url: str) -> bool:
    """Exclude account/private routes and binary/download URLs from public HTML crawling."""
    path = unquote(urlsplit(url).path).lower()
    blocked = (
        "/login",
        "/admin",
        "/account",
        "/logout",
        "/cart",
        "/checkout",
        "/api",
        "/wp-admin",
    )
    binary = (
        ".pdf",
        ".zip",
        ".png",
        ".jpg",
        ".jpeg",
        ".webp",
        ".svg",
        ".mp4",
        ".css",
        ".js",
        ".xml",
        ".gz",
    )
    return not path.startswith(blocked) and not path.endswith(binary)


def split_text(text: str, size: int, overlap: int) -> list[str]:
    """Use LangChain's recursive splitter to keep nearby sentences together with overlap."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", "। ", " ", ""],
        keep_separator=True,
    )
    return splitter.split_text(normalize(text))


def keyword_expression(query: str) -> str:
    """Quote literal tokens so user text cannot inject SQLite FTS query operators."""
    stop = {
        "the",
        "a",
        "an",
        "is",
        "are",
        "can",
        "how",
        "for",
        "to",
        "of",
        "in",
        "my",
        "what",
        "and",
        "i",
    }
    words = [w for w in re.findall(r"[^\W_]+", query.lower(), re.UNICODE) if w not in stop]
    return " OR ".join('"' + w + '"' for w in dict.fromkeys(words[:24]))


def rrf(rankings: list[list[str]], k: int = 60) -> list[str]:
    """Fuse semantic and keyword ranks without treating incomparable raw scores as confidence."""
    scores = defaultdict(float)
    for ranking in rankings:
        for rank, item in enumerate(dict.fromkeys(ranking), 1):
            scores[item] += 1 / (k + rank)
    return sorted(scores, key=lambda item: (-scores[item], item))


def fit_messages(messages: list[dict], budget: int) -> list[dict]:
    """Retain a suffix of complete user/assistant pairs that fits the history token budget."""
    pairs = [messages[i : i + 2] for i in range(0, len(messages), 2)]
    kept, total = [], 0
    for pair in reversed(pairs):
        count = sum(token_count(m["content"]) + 8 for m in pair)
        if total + count > budget:
            break
        kept[0:0] = pair
        total += count
    return kept
