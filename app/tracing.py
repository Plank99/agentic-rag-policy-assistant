"""Tiny shared helper for building trace entries used by both graphs and the Streamlit UI."""
from __future__ import annotations

import time


def trace_step(state: dict, node: str, detail: str, start: float) -> dict:
    step = {"node": node, "detail": detail, "duration_ms": round((time.time() - start) * 1000, 1)}
    return {"trace": state.get("trace", []) + [step]}
