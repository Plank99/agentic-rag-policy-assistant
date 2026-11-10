"""Functional evaluation harness.

Runs every question in eval_questions.json through the real, Ollama-backed agent graph and
automatically grades each answer, then writes a markdown + JSON report to eval/results/.

Grading methodology (documented limitation, see README): this uses cheap, deterministic
checks (keyword containment, numeric tolerance, category/route matching) rather than an
LLM-as-judge, so results are 100% reproducible but will not catch every phrasing of a
correct answer, nor every subtle factual error. A production version would add an
LLM-as-judge pass for the keyword-graded questions.

Usage:
    python -m eval.run_eval
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.graph import build_graph  # noqa: E402

QUESTIONS_PATH = Path(__file__).parent / "eval_questions.json"
RESULTS_DIR = Path(__file__).parent / "results"
TOLERANCE = 0.01


def grade(question: dict, state: dict) -> tuple[bool, str]:
    qtype = question["type"]
    results = state.get("results", {})
    final_answer = (state.get("final_answer") or "").lower()
    rag_answer = (results.get("policy_question", {}).get("answer") or "").lower()
    combined_text = f"{final_answer}\n{rag_answer}"

    if qtype == "out_of_scope":
        nodes = [t["node"] for t in state.get("trace", [])]
        if "direct_response" in nodes and not results:
            return True, "routed to direct_response with no sub-task results, as expected"
        return False, f"expected direct_response short-circuit, got nodes={nodes}"

    if qtype == "leave_balance" or qtype == "compound":
        keyword_ok = True
        keyword_detail = ""
        if "expected_keywords" in question:
            found = [kw for kw in question["expected_keywords"] if kw.lower() in combined_text]
            keyword_ok = len(found) > 0
            keyword_detail = f"keywords found={found}"

        numeric_ok = True
        numeric_detail = ""
        if "expected_remaining_days" in question:
            data = results.get("leave_balance", {}).get("data")
            if not data:
                numeric_ok = False
                numeric_detail = "no leave_balance result produced"
            else:
                actual = data.get("remaining_days")
                expected = question["expected_remaining_days"]
                numeric_ok = actual is not None and abs(actual - expected) <= TOLERANCE
                numeric_detail = f"expected={expected}, actual={actual}"

        category_ok = True
        category_detail = ""
        if "expected_category" in question:
            ticket = results.get("create_ticket")
            category_ok = bool(ticket) and ticket.get("category") == question["expected_category"]
            category_detail = f"expected_category={question['expected_category']}, ticket={ticket}"

        passed = keyword_ok and numeric_ok and category_ok
        detail = "; ".join(d for d in (keyword_detail, numeric_detail, category_detail) if d)
        return passed, detail

    if qtype == "create_ticket":
        ticket = results.get("create_ticket")
        ok = bool(ticket) and ticket.get("category") == question.get("expected_category")
        return ok, f"expected_category={question.get('expected_category')}, ticket={ticket}"

    if qtype == "policy_question":
        found = [kw for kw in question.get("expected_keywords", []) if kw.lower() in combined_text]
        ok = len(found) > 0
        return ok, f"keywords found={found} (looked for {question.get('expected_keywords')})"

    return False, f"unknown question type: {qtype}"


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    questions = json.loads(QUESTIONS_PATH.read_text(encoding="utf-8"))

    print("Building agent graph (loading/building the FAISS index and connecting to Ollama)...")
    graph = build_graph()

    rows = []
    node_durations: dict[str, list[float]] = {}

    for q in questions:
        start = time.time()
        state = graph.invoke({"user_query": q["query"]})
        total_ms = round((time.time() - start) * 1000, 1)
        passed, detail = grade(q, state)

        for step in state.get("trace", []):
            node_durations.setdefault(step["node"], []).append(step["duration_ms"])

        rows.append(
            {
                "id": q["id"],
                "type": q["type"],
                "query": q["query"],
                "passed": passed,
                "detail": detail,
                "total_ms": total_ms,
                "final_answer": state.get("final_answer"),
            }
        )
        print(f"[{'PASS' if passed else 'FAIL'}] {q['id']} ({q['type']}) - {total_ms} ms")

    accuracy = sum(1 for r in rows if r["passed"]) / len(rows)
    node_summary = {
        node: {
            "count": len(durs),
            "avg_ms": round(statistics.mean(durs), 1),
            "max_ms": round(max(durs), 1),
        }
        for node, durs in sorted(node_durations.items())
    }

    report = {
        "accuracy": round(accuracy, 3),
        "num_questions": len(rows),
        "num_passed": sum(1 for r in rows if r["passed"]),
        "node_latency_ms": node_summary,
        "rows": rows,
    }

    (RESULTS_DIR / "eval_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    md_lines = [
        "# Functional Evaluation Report",
        "",
        f"**Accuracy: {report['num_passed']}/{report['num_questions']} "
        f"({accuracy * 100:.1f}%)**",
        "",
        "## Per-node latency (ms)",
        "",
        "| node | count | avg_ms | max_ms |",
        "|---|---|---|---|",
    ]
    for node, stats in node_summary.items():
        md_lines.append(f"| {node} | {stats['count']} | {stats['avg_ms']} | {stats['max_ms']} |")

    md_lines += ["", "## Per-question results", "", "| id | type | pass | total_ms | detail |", "|---|---|---|---|---|"]
    for r in rows:
        status = "PASS" if r["passed"] else "FAIL"
        detail = r["detail"].replace("|", "/")
        md_lines.append(f"| {r['id']} | {r['type']} | {status} | {r['total_ms']} | {detail} |")

    (RESULTS_DIR / "eval_report.md").write_text("\n".join(md_lines), encoding="utf-8")
    print(f"\nAccuracy: {report['num_passed']}/{report['num_questions']} ({accuracy * 100:.1f}%)")
    print(f"Report written to {RESULTS_DIR / 'eval_report.md'} and eval_report.json")


if __name__ == "__main__":
    main()
