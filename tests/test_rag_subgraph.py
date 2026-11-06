"""Exercises the real embedding + FAISS retrieval path, with a scripted LLM standing in for
grading/rewriting/generation so the test is deterministic and needs no Ollama instance."""
from __future__ import annotations

from app.rag.subgraph import build_rag_subgraph, run_rag


def test_happy_path_answers_without_rewrite(tiny_store, scripted_llm):
    llm = scripted_llm(
        [
            '{"relevant": true}',
            "Employees accrue 1.75 days per month (pto.md).",
        ]
    )
    subgraph = build_rag_subgraph(llm, tiny_store, top_k=2, max_rewrites=2)

    result = run_rag(subgraph, "How much PTO do I accrue per month?")

    assert result["confidence"] == "high"
    assert "1.75" in result["answer"]
    assert "pto.md" in result["sources"]
    node_sequence = [t["node"] for t in result["trace"]]
    assert node_sequence == ["retrieve", "grade_relevance", "generate_answer"]


def test_rewrite_loop_then_succeeds(tiny_store, scripted_llm):
    llm = scripted_llm(
        [
            '{"relevant": false}',
            '{"rewritten_query": "PTO monthly accrual rate"}',
            '{"relevant": true}',
            "Accrual is 1.75 days per month (pto.md).",
        ]
    )
    subgraph = build_rag_subgraph(llm, tiny_store, top_k=2, max_rewrites=2)

    result = run_rag(subgraph, "how many days do I get")

    assert result["confidence"] == "high"
    node_sequence = [t["node"] for t in result["trace"]]
    assert node_sequence == [
        "retrieve",
        "grade_relevance",
        "rewrite_query",
        "retrieve",
        "grade_relevance",
        "generate_answer",
    ]


def test_exhausted_rewrites_falls_back_to_low_confidence(tiny_store, scripted_llm):
    llm = scripted_llm(
        [
            '{"relevant": false}',
            '{"rewritten_query": "q2"}',
            '{"relevant": false}',
            '{"rewritten_query": "q3"}',
            '{"relevant": false}',
        ]
    )
    subgraph = build_rag_subgraph(llm, tiny_store, top_k=2, max_rewrites=2)

    result = run_rag(subgraph, "an unanswerable question")

    assert result["confidence"] == "low"
    # retrieve -> grade -> rewrite -> retrieve -> grade -> rewrite -> retrieve -> grade -> generate
    assert result["trace"][-1]["node"] == "generate_answer"
    rewrite_count = sum(1 for t in result["trace"] if t["node"] == "rewrite_query")
    assert rewrite_count == 2
