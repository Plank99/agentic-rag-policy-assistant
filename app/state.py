"""Shared state schemas for the main agent graph and the RAG subgraph."""
from __future__ import annotations

from typing import Any, Optional, TypedDict


class TraceStep(TypedDict):
    node: str
    detail: str
    duration_ms: float


class AgentState(TypedDict, total=False):
    # Input
    user_query: str

    # Planning / decomposition (filled by classify_intent)
    pending_tasks: list[str]          # ordered queue of task types still to execute
    rag_query: str                    # query to send into the RAG subgraph
    calc_args: dict[str, Any]         # extracted args for leave_balance_calculator
    ticket_args: dict[str, Any]       # extracted args for create_support_ticket
    is_out_of_scope: bool

    # Intermediate results, keyed by task type
    results: dict[str, Any]

    # Autonomous escalation flag, set by rag_answer on low-confidence RAG result
    escalate: bool
    escalate_reason: str

    # Observability
    trace: list[TraceStep]

    # Output
    final_answer: str


class RagState(TypedDict, total=False):
    query: str
    original_query: str
    retries: int
    max_retries: int
    docs: list[dict[str, Any]]        # [{"content": ..., "source": ..., "score": ...}]
    relevant: bool
    answer: str
    confidence: str                   # "high" | "low"
    sources: list[str]
    trace: list[TraceStep]
