"""CLI entry point: inspect configuration, index your site, and test persistent chat."""

import argparse
import json
import sys
import uuid
from importlib.metadata import version
from pathlib import Path

from rag.config import ENV_FILE, load_settings
from rag.runtime import close_runtime, runtime
from rag.store import create_store


def output(value):
    """Print readable Unicode JSON for local diagnostics and scripting."""
    print(json.dumps(value, ensure_ascii=False, indent=2))


def doctor():
    """Explain active configuration safely, including .env naming/path and local/server selection."""
    s = load_settings()
    output(
        {
            "python": sys.version.split()[0],
            "env_file": str(ENV_FILE),
            "env_exists": ENV_FILE.exists(),
            "env_txt_exists": ENV_FILE.with_name(".env.txt").exists(),
            "openai_key": "configured"
            if s.openai_key and s.openai_key != "replace-me"
            else "missing/placeholder",
            "backend_key": "configured"
            if len(s.backend_key) >= 32 and not s.backend_key.startswith("replace-")
            else "not set (needed only for API)",
            "qdrant_mode": s.qdrant_mode,
            "qdrant_target": s.qdrant_path
            if s.qdrant_mode == "local"
            else s.qdrant_url,
            "collection": s.collection,
            "db_backend": s.db_backend,
            "database": (
                {
                    "host": s.mysql_host,
                    "port": s.mysql_port,
                    "name": s.mysql_database,
                    "tls": bool(s.mysql_ssl_ca),
                }
                if s.db_backend == "mysql"
                else s.db_path
            ),
            "sqlite_source": s.db_path,
            "models": {
                "chat": s.chat_model,
                "embedding": s.embedding_model,
                "dimensions": s.dimensions,
            },
            "packages": {
                p: version(p)
                for p in (
                    "langchain-core",
                    "langchain-openai",
                    "langchain-qdrant",
                    "qdrant-client",
                )
            },
            "note": "Environment variables override .env. Local mode needs no Docker. This command makes no provider calls.",
        }
    )


def interactive(agent, store, user, conversation_id=None):
    """Keep the same conversation across terminal questions and print its resume UUID."""
    cid = store.conversation(user, conversation_id)["id"]
    print(
        f"Conversation: {cid}\nType /exit to finish. Use --conversation with this UUID to resume."
    )
    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return
        if question == "/exit":
            return
        if not question:
            continue
        if len(question) > 2000:
            print("Please keep the question under 2000 characters.")
            continue
        result = agent.respond(user, cid, question, str(uuid.uuid4()))
        print("Bot:", result["answer"])
        for source in result["sources"]:
            print(f"  [{source['id']}] {source['url']}")


def build_parser():
    """Define a small command surface; parse and validate inputs before loading provider clients."""
    parser = argparse.ArgumentParser(description="StartupFlora LangChain RAG")
    subs = parser.add_subparsers(dest="command", required=True)
    subs.add_parser(
        "doctor", help="Show safe configuration diagnostics; no API key needed"
    )
    subs.add_parser("status", help="List indexed sources and chunk counts")
    subs.add_parser(
        "cleanup-vectors",
        help="Remove orphaned vector generations from the dedicated corpus",
    )
    web = subs.add_parser(
        "ingest-web", help="Discover sitemap/internal links and index approved pages"
    )
    web.add_argument("--url", default="https://startupflora.com/")
    web.add_argument(
        "--max-pages",
        type=int,
        default=0,
        help="0 = process discovered pages up to MAX_CRAWL_URLS",
    )
    web.add_argument(
        "--sitemap", help="Optional explicit HTTPS sitemap URL on an allowed host"
    )
    web.add_argument("--report", default="data/v2/crawl-report.json")
    pdf = subs.add_parser("ingest-pdf", help="Index an approved public text PDF")
    pdf.add_argument("file")
    pdf.add_argument("--source-url", required=True)
    for name in ("ask", "chat"):
        command = subs.add_parser(
            name,
            help="Ask once"
            if name == "ask"
            else "Interactive persistent terminal chat",
        )
        if name == "ask":
            command.add_argument("question")
        command.add_argument("--user", default="local-demo")
        command.add_argument("--conversation")
    remove = subs.add_parser("remove-source", help="Remove one obsolete indexed source")
    remove.add_argument("url")
    cleanup = subs.add_parser(
        "cleanup-cache", help="Delete old embedding/search cache entries"
    )
    cleanup.add_argument("--days", type=int, default=30)
    migrate = subs.add_parser(
        "migrate-history",
        help="Copy conversations from a previous DB without its old vectors",
    )
    migrate.add_argument("--from-db", required=True)
    return parser


def main():
    """Route CLI operations and always close local Qdrant before another process uses it."""
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "ingest-web" and args.max_pages < 0:
        parser.error("--max-pages must be 0 or a positive number")
    if args.command == "ask" and not 1 <= len(args.question.strip()) <= 2000:
        parser.error("Question must be 1-2000 characters")
    if args.command == "cleanup-cache" and args.days < 1:
        parser.error("--days must be positive")
    try:
        if args.command == "doctor":
            doctor()
            return
        if args.command in {"status", "cleanup-cache", "migrate-history"}:
            s = load_settings()
            store = create_store(s)
            if args.command == "status":
                sources = store.inventory()
                output(
                    {
                        "sources": sources,
                        "documents": len(sources),
                        "revision": store.revision(),
                    }
                )
            elif args.command == "cleanup-cache":
                output(store.cleanup_cache(args.days))
            else:
                output({"conversations_imported": store.import_history(args.from_db)})
            return
        from rag.ingest import Ingestor

        s, store, models, retriever, agent = runtime()
        ingestor = Ingestor(s, store, models, retriever)
        if args.command == "ingest-web":
            report = ingestor.crawl(args.url, args.max_pages, args.sitemap)
            report_path = Path(args.report)
            report_path.parent.mkdir(parents=True, exist_ok=True)
            report_path.write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            summary = {
                k: v
                for k, v in report.items()
                if k not in {"outcomes", "errors", "pending_urls_first_200"}
            }
            output(
                {
                    "report": str(report_path.resolve()),
                    **summary,
                    "errors_first_5": report["errors"][:5],
                }
            )
            if report["indexed"] + report["unchanged"] == 0:
                raise SystemExit(
                    "No pages indexed. Review the report, robots rules, networking or extracted HTML."
                )
        elif args.command == "ingest-pdf":
            output(ingestor.pdf(args.file, args.source_url))
        elif args.command == "cleanup-vectors":
            output(ingestor.cleanup_vectors())
        elif args.command == "remove-source":
            output(ingestor.remove_source(args.url))
        elif args.command == "chat":
            interactive(agent, store, args.user, args.conversation)
        else:
            cid = store.conversation(args.user, args.conversation)["id"]
            output(
                agent.respond(args.user, cid, args.question.strip(), str(uuid.uuid4()))
            )
    except (ValueError, FileNotFoundError) as exc:
        # Settings hide sensitive input values in their validation errors.
        raise SystemExit(str(exc)) from None
    finally:
        close_runtime()


if __name__ == "__main__":
    main()
