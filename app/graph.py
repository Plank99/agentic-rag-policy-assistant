"""Main agentic workflow.

Six nodes (the dedicated RAG subgraph in app/rag/subgraph.py is separate and not counted
here):

  classify_intent     - entry point; LLM decomposes the query into an ordered task queue
                        (autonomous decision + subtask decomposition)
  direct_response     - short-circuit for greetings / out-of-scope messages
  rag_answer          - delegates to the RAG subgraph; auto-escalates to a ticket on
                        low-confidence answers (autonomous decision)
  calculate_leave     - deterministic PTO calculator tool
  create_ticket       - support-ticket tool (user-requested or auto-escalated)
  synthesize_response - combines whatever sub-task results were collected into one answer

State (intermediate results, task queue, and a full execution trace) is threaded through
every node via `AgentState` - see app/state.py.
"""
from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from app.llm import get_chat_llm, structured_json_call
from app.rag.subgraph import build_rag_subgraph, run_rag
from app.rag.vectorstore import load_or_build_index
from app.state import AgentState
from app.tools.calculator import compute_leave_balance
from app.tools.ticketing import create_ticket_record
from app.tracing import trace_step as _trace

TASK_NODE_MAP = {
    "policy_question": "rag_answer",
    "leave_balance": "calculate_leave",
    "create_ticket": "create_ticket",
}

_ROUTE_TARGETS = {node: node for node in set(TASK_NODE_MAP.values()) | {"synthesize_response"}}

_CLASSIFY_PROMPT = """You are the routing brain of an internal company-policy assistant. \
Decompose the user's message into an ORDERED list of task types drawn from:
  - "policy_question": needs to look up something in internal HR/IT/Finance/Legal policy docs
  - "leave_balance": needs to calculate remaining PTO/vacation balance (requires hire_date \
and days_taken)
  - "create_ticket": the user explicitly wants to open a support/HR/IT ticket

If the message is a greeting, small talk, or unrelated to company policy/HR/IT, return an \
empty task list.

Respond with ONLY a JSON object of this shape:
{{
  "tasks": ["policy_question", "leave_balance"],
  "rag_query": "self-contained question to search policy docs for, or empty string",
  "calc_args": {{"hire_date": "YYYY-MM-DD", "days_taken": 0, "as_of_date": "YYYY-MM-DD or omit"}},
  "ticket_args": {{"category": "short_snake_case_category", "summary": "...", "priority": "low|normal|high"}}
}}
Only include calc_args if "leave_balance" is in tasks, and only include ticket_args if \
"create_ticket" is in tasks. A single message can contain multiple tasks.

User message: {query}
"""

_DIRECT_RESPONSE_PROMPT = """The user said: {query!r}
This assistant only answers questions about internal company policy (HR, IT, Finance, \
Legal), calculates PTO/vacation balances, and opens support tickets. Reply briefly (1-3 \
sentences) and politely: greet back if it's a greeting, otherwise explain you can't help \
with that and suggest what you *can* help with.
"""

_SYNTHESIZE_PROMPT = """Combine the following internal results into ONE clear, friendly \
answer for the employee who asked the original question below. Preserve any source \
document names and ticket IDs mentioned in the results. Do not invent facts that are not \
in the results.

Original question: {query}

Results:
{context}
"""


def _classify_intent(llm):
    def node(state: AgentState) -> dict:
        start = time.time()
        default = {
            "tasks": ["policy_question"],
            "rag_query": state["user_query"],
            "calc_args": {},
            "ticket_args": {},
        }
        result = structured_json_call(llm, _CLASSIFY_PROMPT.format(query=state["user_query"]), default)
        tasks = [t for t in result.get("tasks", []) if t in TASK_NODE_MAP]
        update = {
            "pending_tasks": tasks,
            "rag_query": result.get("rag_query") or state["user_query"],
            "calc_args": result.get("calc_args", {}) or {},
            "ticket_args": result.get("ticket_args", {}) or {},
            "results": {},
            "escalate": False,
        }
        update.update(_trace(state, "classify_intent", f"tasks={tasks}", start))
        return update

    return node


def _direct_response(llm):
    def node(state: AgentState) -> dict:
        start = time.time()
        response = llm.invoke(_DIRECT_RESPONSE_PROMPT.format(query=state["user_query"]))
        answer = response.content if hasattr(response, "content") else str(response)
        update = {"final_answer": answer}
        update.update(_trace(state, "direct_response", "out-of-scope / greeting", start))
        return update

    return node


def _rag_answer(rag_subgraph):
    def node(state: AgentState) -> dict:
        start = time.time()
        query = state.get("rag_query") or state["user_query"]
        rag_result = run_rag(rag_subgraph, query)

        results = dict(state.get("results", {}))
        results["policy_question"] = {
            "answer": rag_result.get("answer", ""),
            "sources": rag_result.get("sources", []),
            "confidence": rag_result.get("confidence", "low"),
        }

        remaining = list(state.get("pending_tasks", []))[1:]
        escalate = False
        escalate_reason = ""
        ticket_args = state.get("ticket_args", {})
        if rag_result.get("confidence") == "low" and "create_ticket" not in remaining:
            remaining = remaining + ["create_ticket"]
            escalate = True
            escalate_reason = "RAG subgraph could not confidently answer the policy question."
            ticket_args = {
                "category": "general_policy_question",
                "summary": f"Could not confidently answer policy question: {query}",
                "priority": "normal",
            }

        sub_trace = [
            {"node": f"rag.{t['node']}", "detail": t["detail"], "duration_ms": t["duration_ms"]}
            for t in rag_result.get("trace", [])
        ]

        update = {
            "results": results,
            "pending_tasks": remaining,
            "escalate": escalate,
            "escalate_reason": escalate_reason,
            "ticket_args": ticket_args,
            "trace": state.get("trace", []) + sub_trace,
        }
        update.update(_trace(update, "rag_answer", f"confidence={rag_result.get('confidence')}", start))
        return update

    return node


def _calculate_leave():
    def node(state: AgentState) -> dict:
        start = time.time()
        args = state.get("calc_args") or {}
        try:
            data = compute_leave_balance(**args)
            summary = (
                f"Accrued {data['accrued_days']} days, already taken {data['days_taken']}, "
                f"remaining {data['remaining_days']} days (as of {data['as_of_date']})."
            )
            error = None
        except Exception as exc:  # noqa: BLE001 - deliberately broad: surfaces to the user
            data = None
            error = str(exc)
            summary = f"Could not compute leave balance: missing or invalid input ({error})."

        results = dict(state.get("results", {}))
        results["leave_balance"] = {"data": data, "summary": summary, "error": error}
        remaining = list(state.get("pending_tasks", []))[1:]
        update = {"results": results, "pending_tasks": remaining}
        update.update(_trace(state, "calculate_leave", summary, start))
        return update

    return node


def _create_ticket():
    def node(state: AgentState) -> dict:
        start = time.time()
        args = state.get("ticket_args") or {}
        category = args.get("category", "general_inquiry")
        summary = args.get("summary", state["user_query"])
        priority = args.get("priority", "normal")
        ticket = create_ticket_record(
            category=category, summary=summary, user_query=state["user_query"], priority=priority
        )
        results = dict(state.get("results", {}))
        results["create_ticket"] = ticket
        remaining = list(state.get("pending_tasks", []))[1:]
        update = {"results": results, "pending_tasks": remaining}
        update.update(
            _trace(state, "create_ticket", f"opened {ticket['ticket_id']} ({category})", start)
        )
        return update

    return node


def _synthesize_response(llm):
    def node(state: AgentState) -> dict:
        start = time.time()
        results = state.get("results", {})
        parts = []
        if "policy_question" in results:
            r = results["policy_question"]
            parts.append(
                f"Policy answer (confidence={r['confidence']}, sources={r['sources']}):\n{r['answer']}"
            )
        if "leave_balance" in results:
            parts.append(f"Leave balance calculation:\n{results['leave_balance']['summary']}")
        if "create_ticket" in results:
            t = results["create_ticket"]
            parts.append(
                f"Support ticket created: {t['ticket_id']} "
                f"(category={t['category']}, priority={t['priority']}, status={t['status']})."
            )
        context = "\n\n".join(parts) if parts else "No sub-task results were produced."

        response = llm.invoke(
            _SYNTHESIZE_PROMPT.format(query=state["user_query"], context=context)
        )
        answer = response.content if hasattr(response, "content") else str(response)
        update = {"final_answer": answer}
        update.update(_trace(state, "synthesize_response", "combined final answer", start))
        return update

    return node


def _route_after_classify(state: AgentState) -> str:
    tasks = state.get("pending_tasks") or []
    if not tasks:
        return "direct_response"
    return TASK_NODE_MAP[tasks[0]]


def _route_dispatch(state: AgentState) -> str:
    tasks = state.get("pending_tasks") or []
    if tasks:
        return TASK_NODE_MAP[tasks[0]]
    return "synthesize_response"


def build_graph(llm=None, store=None):
    """Compile the main agent graph. Pass `llm`/`store` explicitly to inject fakes in tests."""
    llm = llm or get_chat_llm()
    store = store or load_or_build_index()
    rag_subgraph = build_rag_subgraph(llm, store)

    graph = StateGraph(AgentState)
    graph.add_node("classify_intent", _classify_intent(llm))
    graph.add_node("direct_response", _direct_response(llm))
    graph.add_node("rag_answer", _rag_answer(rag_subgraph))
    graph.add_node("calculate_leave", _calculate_leave())
    graph.add_node("create_ticket", _create_ticket())
    graph.add_node("synthesize_response", _synthesize_response(llm))

    graph.set_entry_point("classify_intent")
    graph.add_conditional_edges(
        "classify_intent",
        _route_after_classify,
        {"direct_response": "direct_response", **_ROUTE_TARGETS},
    )
    for source in ("rag_answer", "calculate_leave", "create_ticket"):
        graph.add_conditional_edges(source, _route_dispatch, _ROUTE_TARGETS)

    graph.add_edge("direct_response", END)
    graph.add_edge("synthesize_response", END)

    return graph.compile()
