"""Evaluate retrieval against reviewed source URLs. This invokes your configured paid models."""

import argparse
import json
import time
from pathlib import Path

from rag.runtime import close_runtime, runtime


def evaluate(cases_path):
    """Measure source recall/abstention and latency; do not present these as answer-accuracy scores."""
    cases = [
        json.loads(line)
        for line in Path(cases_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not cases:
        raise ValueError("Evaluation file is empty")
    for case in cases:
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError("Each case requires a query")
        if "REPLACE" in json.dumps(case):
            raise ValueError("Replace example source URLs with manually reviewed URLs first")
        if not case.get("should_abstain") and not case.get("expected_sources"):
            raise ValueError("Non-abstention cases require reviewed expected_sources")
    rows = []
    _, _, _, retriever, _ = runtime()
    for case in cases:
        start = time.monotonic()
        chunks = retriever.search(case["query"])
        found = {c["source"] for c in chunks}
        expected = set(case.get("expected_sources", []))
        rows.append(
            {
                "query": case["query"],
                "retrieved_sources": sorted(found),
                "source_recall": len(found & expected) / len(expected) if expected else None,
                "abstention_correct": not bool(found) if case.get("should_abstain") else None,
                "elapsed_ms": round((time.monotonic() - start) * 1000),
            }
        )
    scores = [r["source_recall"] for r in rows if r["source_recall"] is not None]
    return {
        "cases": rows,
        "mean_source_recall": sum(scores) / len(scores) if scores else None,
        "note": "Retrieval-only evaluation; model grounding and final answers need separate review. Warm caches affect latency.",
    }


def main():
    """Write an evaluation report and always release the local Qdrant storage lock."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", required=True)
    parser.add_argument("--output", default="data/v2/evaluation.json")
    args = parser.parse_args()
    try:
        result = evaluate(args.cases)
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(
            json.dumps(
                {
                    "report": str(output.resolve()),
                    "mean_source_recall": result["mean_source_recall"],
                },
                indent=2,
            )
        )
    finally:
        close_runtime()


if __name__ == "__main__":
    main()
