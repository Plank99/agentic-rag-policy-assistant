"""Deterministic PTO (leave balance) calculator tool.

This is intentionally plain Python (no LLM call) wrapped as a LangChain tool: the agent
extracts structured arguments from the user's message with an LLM, but the arithmetic itself
is deterministic and auditable, which matters for an HR-facing calculation.

Accrual rule, mirrored from app/data/policies/pto_policy.md:
  - 1.75 days accrued per full calendar month of employment (21 days/year)
  - max carryover of 5 days is ignored here (single-year, from-hire-date balance); a
    production version would need full year-boundary bookkeeping.
"""
from __future__ import annotations

from datetime import date

from langchain_core.tools import tool

ACCRUAL_DAYS_PER_MONTH = 1.75


def _months_employed(hire_date: date, as_of: date) -> int:
    if as_of < hire_date:
        return 0
    months = (as_of.year - hire_date.year) * 12 + (as_of.month - hire_date.month)
    if as_of.day < hire_date.day:
        months -= 1
    return max(months, 0)


def compute_leave_balance(hire_date: str, days_taken: float, as_of_date: str | None = None) -> dict:
    """Compute accrued and remaining PTO balance.

    Args:
        hire_date: ISO date string (YYYY-MM-DD) the employee was hired.
        days_taken: Number of PTO days already used this year.
        as_of_date: ISO date string to calculate the balance as of; defaults to today.
    """
    hire = date.fromisoformat(hire_date)
    as_of = date.fromisoformat(as_of_date) if as_of_date else date.today()

    months = _months_employed(hire, as_of)
    accrued = round(months * ACCRUAL_DAYS_PER_MONTH, 2)
    remaining = round(accrued - float(days_taken), 2)

    return {
        "hire_date": hire.isoformat(),
        "as_of_date": as_of.isoformat(),
        "months_employed": months,
        "accrued_days": accrued,
        "days_taken": float(days_taken),
        "remaining_days": remaining,
    }


@tool
def leave_balance_calculator(hire_date: str, days_taken: float, as_of_date: str | None = None) -> dict:
    """Calculate an employee's remaining PTO balance from hire date and days already taken.

    hire_date and as_of_date must be ISO format strings (YYYY-MM-DD). as_of_date defaults to
    today if omitted.
    """
    return compute_leave_balance(hire_date, days_taken, as_of_date)
