"""Verifies the 6-node main graph's conditional routing, task decomposition, and the
autonomous RAG-failure -> auto-escalation-ticket behavior, using a scripted LLM so the test
needs no running Ollama instance."""
from __future__ import annotations

from app.graph import build_graph


def test_out_of_scope_short_circuits_to_direct_response(scripted_llm, tiny_store):
    llm = scripted_llm(
        [
            '{"tasks": []}',
            "Hi there! I can help with company policy questions, PTO balances, and tickets.",
        ]
    )
    graph = build_graph(llm=llm, store=tiny_store)

    result = graph.invoke({"user_query": "hey, how's it going?"})

    nodes = [t["node"] for t in result["trace"]]
    assert nodes == ["classify_intent", "direct_response"]
    assert "final_answer" in result


def test_policy_question_routes_through_rag_and_synthesize(scripted_llm, tiny_store):
    llm = scripted_llm(
        [
            '{"tasks": ["policy_question"], "rag_query": "PTO monthly accrual"}',
            '{"relevant": true}',
            "Accrual is 1.75 days per month (pto.md).",
            "You accrue 1.75 PTO days per month.",
        ]
    )
    graph = build_graph(llm=llm, store=tiny_store)

    result = graph.invoke({"user_query": "How much PTO do I accrue per month?"})

    nodes = [t["node"] for t in result["trace"]]
    assert "classify_intent" in nodes
    assert "rag_answer" in nodes
    assert nodes[-1] == "synthesize_response"
    assert result["results"]["policy_question"]["confidence"] == "high"
    assert not result["escalate"]


def test_leave_balance_routes_through_calculator(scripted_llm, tiny_store):
    llm = scripted_llm(
        [
            '{"tasks": ["leave_balance"], "calc_args": {"hire_date": "2024-01-01", '
            '"days_taken": 2, "as_of_date": "2024-07-01"}}',
            "You have accrued 10.5 days and taken 2, leaving 8.5 days.",
        ]
    )
    graph = build_graph(llm=llm, store=tiny_store)

    result = graph.invoke({"user_query": "How many PTO days do I have left?"})

    nodes = [t["node"] for t in result["trace"]]
    assert nodes == ["classify_intent", "calculate_leave", "synthesize_response"]
    assert result["results"]["leave_balance"]["data"]["remaining_days"] == 8.5


def test_create_ticket_task_opens_a_ticket(scripted_llm, tiny_store, tickets_db):
    llm = scripted_llm(
        [
            '{"tasks": ["create_ticket"], "ticket_args": {"category": "it_access_request", '
            '"summary": "Need VPN access", "priority": "high"}}',
            "I've opened a high-priority IT ticket for your VPN access request.",
        ]
    )
    graph = build_graph(llm=llm, store=tiny_store)

    result = graph.invoke({"user_query": "I need VPN access as a contractor."})

    nodes = [t["node"] for t in result["trace"]]
    assert nodes == ["classify_intent", "create_ticket", "synthesize_response"]
    ticket = result["results"]["create_ticket"]
    assert ticket["category"] == "it_access_request"
    assert ticket["priority"] == "high"
    assert ticket["ticket_id"].startswith("TCK-")


def test_compound_query_executes_both_subtasks_in_order(scripted_llm, tiny_store):
    llm = scripted_llm(
        [
            '{"tasks": ["policy_question", "leave_balance"], "rag_query": "PTO accrual rate", '
            '"calc_args": {"hire_date": "2024-01-01", "days_taken": 0, "as_of_date": "2024-07-01"}}',
            '{"relevant": true}',
            "Accrual is 1.75 days per month (pto.md).",
            "Combined answer: 1.75/month accrual, 10.5 days remaining.",
        ]
    )
    graph = build_graph(llm=llm, store=tiny_store)

    result = graph.invoke({"user_query": "What's my PTO accrual rate and how many days do I have left?"})

    nodes = [t["node"] for t in result["trace"]]
    assert nodes == [
        "classify_intent",
        "rag.retrieve",
        "rag.grade_relevance",
        "rag.generate_answer",
        "rag_answer",
        "calculate_leave",
        "synthesize_response",
    ]
    assert "policy_question" in result["results"]
    assert "leave_balance" in result["results"]


def test_low_confidence_rag_auto_escalates_to_ticket(scripted_llm, tiny_store, tickets_db):
    llm = scripted_llm(
        [
            '{"tasks": ["policy_question"], "rag_query": "an unanswerable policy question"}',
            '{"relevant": false}',
            '{"rewritten_query": "q2"}',
            '{"relevant": false}',
            '{"rewritten_query": "q3"}',
            '{"relevant": false}',
            "I couldn't find a confident answer, so I opened a ticket for you.",
        ]
    )
    graph = build_graph(llm=llm, store=tiny_store)

    result = graph.invoke({"user_query": "some obscure policy question"})

    assert result["escalate"] is True
    assert result["results"]["policy_question"]["confidence"] == "low"
    assert "create_ticket" in result["results"]
    nodes = [t["node"] for t in result["trace"]]
    assert nodes[-2:] == ["create_ticket", "synthesize_response"]
