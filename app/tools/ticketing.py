"""Mock support-ticket system.

A real side-effecting (non-retrieval) tool: it writes a row to a local SQLite database and
returns a ticket id, simulating a corporate ticketing system (e.g. ServiceNow/Jira Service
Desk) that the agent can autonomously open on the user's behalf, or as a self-triggered
escalation when the RAG subgraph cannot confidently answer a policy question.
"""
from __future__ import annotations

import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone

from langchain_core.tools import tool

from app.config import TICKETS_DB_PATH

VALID_PRIORITIES = {"low", "normal", "high"}


@contextmanager
def _connect():
    TICKETS_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(TICKETS_DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                ticket_id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                summary TEXT NOT NULL,
                user_query TEXT,
                priority TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def create_ticket_record(
    category: str, summary: str, user_query: str = "", priority: str = "normal"
) -> dict:
    priority = priority if priority in VALID_PRIORITIES else "normal"
    ticket_id = f"TCK-{uuid.uuid4().hex[:8].upper()}"
    created_at = datetime.now(timezone.utc).isoformat()

    with _connect() as conn:
        conn.execute(
            "INSERT INTO tickets (ticket_id, category, summary, user_query, priority, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (ticket_id, category, summary, user_query, priority, "open", created_at),
        )

    return {
        "ticket_id": ticket_id,
        "category": category,
        "summary": summary,
        "priority": priority,
        "status": "open",
        "created_at": created_at,
    }


def get_ticket(ticket_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT ticket_id, category, summary, user_query, priority, status, created_at "
            "FROM tickets WHERE ticket_id = ?",
            (ticket_id,),
        ).fetchone()
    if row is None:
        return None
    keys = ["ticket_id", "category", "summary", "user_query", "priority", "status", "created_at"]
    return dict(zip(keys, row))


@tool
def create_support_ticket(
    category: str, summary: str, user_query: str = "", priority: str = "normal"
) -> dict:
    """Open a support ticket in the corporate ticketing system.

    category: a short machine-readable category, e.g. 'it_access_request',
      'hr_leave_request', 'hr_conduct_concern', 'finance_expense_dispute',
      'legal_data_subject_request', 'it_security_incident', 'it_equipment_request',
      'travel_safety_approval'.
    summary: a one or two sentence human-readable description of the request.
    priority: one of 'low', 'normal', 'high'.
    """
    return create_ticket_record(category, summary, user_query, priority)
