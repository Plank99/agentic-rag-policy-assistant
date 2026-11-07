"""Simplified load test: fires a batch of synthetic queries at the real agent graph and
reports latency percentiles, per-node timing, and a bottleneck analysis.

This simulates mixed traffic across the four intent types in roughly realistic proportions
(mostly policy questions, some calculator/ticket requests, a few chit-chat/out-of-scope
messages and compound questions) by cycling through template queries. 150 real queries
against a local Ollama model takes a while on CPU - that is itself part of the point of
this exercise: see the bottleneck analysis in the generated report.

Usage:
    python -m eval.load_test                 # 150 queries, sequential
    python -m eval.load_test --num 60         # fewer queries, for a quick run
    python -m eval.load_test --workers 4      # fan out with a thread pool
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import build_graph  # noqa: E402

RESULTS_DIR = Path(__file__).parent / "results"

POLICY_TEMPLATES = [
    "What are the password requirements for corporate accounts?",
    "How many PTO days can I carry over into next year?",
    "How many days per week can I work remotely?",
    "What's the receipt threshold for expenses?",
    "How quickly must a GDPR data subject request be answered?",
    "What's the maximum value of a gift I can accept?",
    "How long do I have to return equipment after leaving the company?",
    "What's the nightly hotel cap for business travel in major cities?",
    "Do I need a VPN to work from a cafe?",
    "How often do I need to rotate my password?",
]

LEAVE_TEMPLATES = [
    "I was hired on {hire} and have taken {taken} PTO days. How many do I have left as of {as_of}?",
    "Hired {hire}, took {taken} days off, what's my balance on {as_of}?",
]
LEAVE_PARAMS = [
    ("2021-03-01", 4, "2023-03-01"),
    ("2022-07-15", 2, "2023-01-15"),
    ("2023-10-01", 0, "2024-04-01"),
    ("2020-01-01", 10, "2021-01-01"),
]

TICKET_TEMPLATES = [
    "Please open a ticket, I lost my company laptop.",
    "I need a VPN account set up as a new contractor, can you file a ticket?",
    "Open a ticket, I need to report a lost expense receipt for 80 EUR.",
    "Please file a ticket to report a possible phishing email I received.",
]

OUT_OF_SCOPE_TEMPLATES = [
    "What's the weather like today?",
    "Can you recommend a good pizza place?",
    "Tell me a joke.",
]

COMPOUND_TEMPLATES = [
    "What's the PTO carryover limit, and separately, I was hired {hire} with {taken} days taken - "
    "how many do I have left as of {as_of}?",
]


def build_query_set(num: int) -> list[str]:
    queries: list[str] = []
    i = 0
    while len(queries) < num:
        bucket = i % 10
        if bucket < 5:
            queries.append(POLICY_TEMPLATES[i % len(POLICY_TEMPLATES)])
        elif bucket < 7:
            template = LEAVE_TEMPLATES[i % len(LEAVE_TEMPLATES)]
            hire, taken, as_of = LEAVE_PARAMS[i % len(LEAVE_PARAMS)]
            queries.append(template.format(hire=hire, taken=taken, as_of=as_of))
        elif bucket < 9:
            queries.append(TICKET_TEMPLATES[i % len(TICKET_TEMPLATES)])
        elif bucket == 9:
            if i % 20 == 9:
                hire, taken, as_of = LEAVE_PARAMS[i % len(LEAVE_PARAMS)]
                queries.append(COMPOUND_TEMPLATES[0].format(hire=hire, taken=taken, as_of=as_of))
            else:
                queries.append(OUT_OF_SCOPE_TEMPLATES[i % len(OUT_OF_SCOPE_TEMPLATES)])
        i += 1
    return queries[:num]


def percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    k = (len(ordered) - 1) * (p / 100)
    f, c = int(k), min(int(k) + 1, len(ordered) - 1)
    if f == c:
        return ordered[f]
    return ordered[f] + (ordered[c] - ordered[f]) * (k - f)


def run_load_test(graph, queries: list[str], workers: int = 1) -> dict:
    results = []

    def _run_one(query: str) -> dict:
        start = time.time()
        state = graph.invoke({"user_query": query})
        total_ms = (time.time() - start) * 1000
        return {"query": query, "total_ms": total_ms, "trace": state.get("trace", [])}

    if workers <= 1:
        for q in queries:
            results.append(_run_one(q))
    else:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_run_one, queries))

    return summarize(results)


def summarize(results: list[dict]) -> dict:
    total_latencies = [r["total_ms"] for r in results]
    node_latencies: dict[str, list[float]] = {}
    for r in results:
        for step in r["trace"]:
            node_latencies.setdefault(step["node"], []).append(step["duration_ms"])

    overall = {
        "n": len(results),
        "mean_ms": round(statistics.mean(total_latencies), 1) if total_latencies else 0,
        "p50_ms": round(percentile(total_latencies, 50), 1),
        "p90_ms": round(percentile(total_latencies, 90), 1),
        "p99_ms": round(percentile(total_latencies, 99), 1),
        "max_ms": round(max(total_latencies), 1) if total_latencies else 0,
    }

    by_node = {}
    for node, durs in sorted(node_latencies.items()):
        by_node[node] = {
            "n": len(durs),
            "mean_ms": round(statistics.mean(durs), 1),
            "p50_ms": round(percentile(durs, 50), 1),
            "p90_ms": round(percentile(durs, 90), 1),
            "share_of_total_pct": 0.0,  # filled in below
        }

    total_mean_sum = sum(v["mean_ms"] * v["n"] for v in by_node.values())
    if total_mean_sum > 0:
        for v in by_node.values():
            v["share_of_total_pct"] = round(100 * (v["mean_ms"] * v["n"]) / total_mean_sum, 1)

    bottleneck = max(by_node.items(), key=lambda kv: kv[1]["mean_ms"] * kv[1]["n"], default=(None, None))

    return {"overall": overall, "by_node": by_node, "bottleneck_node": bottleneck[0]}


OPTIMIZATION_NOTES = """
## Bottleneck analysis

The dominant cost is almost always the LLM-calling nodes (`classify_intent`,
`rag.grade_relevance`, `rag.rewrite_query`, `rag.generate_answer`, `synthesize_response`):
every one of them makes a synchronous round-trip to the local Ollama model, and a single
policy question can trigger 3-5 of these round-trips in sequence. `retrieve` (FAISS) and the
deterministic tools (`calculate_leave`, `create_ticket`) are essentially free by comparison
(single-digit milliseconds), since they never leave the process.

## Optimization suggestions

1. **Collapse LLM round-trips.** `grade_relevance` + `rewrite_query` can be merged into a
   single structured-output call (`{"relevant": bool, "rewritten_query": "..."}`), and
   `grade_relevance` can be skipped entirely when the top FAISS similarity score already
   clears a confidence threshold. That alone removes up to 2 of the 4-5 LLM calls per
   request on the happy path.
2. **Reduce per-call model latency.** Use a smaller/quantized model (or Ollama's
   `num_ctx`/`keep_alive` tuning to avoid repeated context reloads), and batch the
   embedding step for `retrieve` across concurrent requests. For real concurrent load,
   front Ollama with more than one model replica (or a GPU), since a single local Ollama
   process serializes requests regardless of how many client threads call it.
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num", type=int, default=150)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    queries = build_query_set(args.num)

    print(f"Building agent graph, then running {len(queries)} queries (workers={args.workers})...")
    graph = build_graph()

    start = time.time()
    summary = run_load_test(graph, queries, workers=args.workers)
    wall_clock_s = round(time.time() - start, 1)
    summary["wall_clock_s"] = wall_clock_s
    summary["workers"] = args.workers

    (RESULTS_DIR / "load_test_report.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md = [
        "# Load Test Report",
        "",
        f"- Queries: {summary['overall']['n']}",
        f"- Workers: {args.workers}",
        f"- Wall clock: {wall_clock_s}s",
        "",
        "## Overall latency (ms)",
        "",
        "| mean | p50 | p90 | p99 | max |",
        "|---|---|---|---|---|",
        f"| {summary['overall']['mean_ms']} | {summary['overall']['p50_ms']} | "
        f"{summary['overall']['p90_ms']} | {summary['overall']['p99_ms']} | {summary['overall']['max_ms']} |",
        "",
        "## Per-node latency (ms)",
        "",
        "| node | n | mean | p50 | p90 | share of total |",
        "|---|---|---|---|---|---|",
    ]
    for node, stats in summary["by_node"].items():
        md.append(
            f"| {node} | {stats['n']} | {stats['mean_ms']} | {stats['p50_ms']} | "
            f"{stats['p90_ms']} | {stats['share_of_total_pct']}% |"
        )
    md.append(f"\n**Identified bottleneck node:** `{summary['bottleneck_node']}`\n")
    md.append(OPTIMIZATION_NOTES)

    (RESULTS_DIR / "load_test_report.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nDone in {wall_clock_s}s. Report written to {RESULTS_DIR / 'load_test_report.md'}")


if __name__ == "__main__":
    main()
