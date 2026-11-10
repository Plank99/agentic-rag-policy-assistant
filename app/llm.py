"""LLM access layer.

Trade-off note (see README for the full discussion): we default to a small local Ollama
model (llama3.2:3b) so the prototype runs on a laptop CPU without any paid API. Small models
are fast but less reliable at emitting strict JSON, so `structured_json_call` below is
deliberately defensive: it asks for JSON, then extracts the first JSON object found in the
response rather than trusting the model to emit *only* JSON.
"""
from __future__ import annotations

import json
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL

_JSON_BLOCK_RE = re.compile(r"\{.*\}", re.DOTALL)


def get_chat_llm() -> BaseChatModel:
    """Build the default local Ollama chat model used by the agent."""
    from langchain_ollama import ChatOllama

    return ChatOllama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL, temperature=0)


def structured_json_call(llm: BaseChatModel, prompt: str, default: dict[str, Any]) -> dict[str, Any]:
    """Invoke the LLM and best-effort parse a JSON object out of its response.

    Falls back to `default` if the model does not return parseable JSON, which keeps the
    graph resilient to small/local models that occasionally ignore formatting instructions.
    """
    response = llm.invoke(prompt)
    text = response.content if hasattr(response, "content") else str(response)
    match = _JSON_BLOCK_RE.search(text)
    if not match:
        return default
    try:
        parsed = json.loads(match.group(0))
    except json.JSONDecodeError:
        return default
    if not isinstance(parsed, dict):
        return default
    merged = dict(default)
    merged.update(parsed)
    return merged
