"""Shared test fixtures.

No Ollama instance is available in CI/this sandbox, so graph-wiring tests inject a small
scripted stub instead of a real chat model. It only needs to support `.invoke(prompt) ->
object with `.content`, which is all `app/llm.py` and the graph nodes rely on - so it is a
faithful stand-in for exercising routing/state logic independent of actual answer quality.
Embedding-based retrieval (`tests/test_rag_subgraph.py`) still uses the real local
sentence-transformers model and a real FAISS index, since that part needs no Ollama.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest


class ScriptedLLM:
    """Returns responses from a fixed list, one per `.invoke()` call, repeating the last."""

    def __init__(self, responses: list[str]):
        self.responses = list(responses)
        self.calls: list[str] = []
        self._i = 0

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        if self._i < len(self.responses):
            text = self.responses[self._i]
        else:
            text = self.responses[-1] if self.responses else "{}"
        self._i += 1
        return SimpleNamespace(content=text)


@pytest.fixture
def scripted_llm():
    return ScriptedLLM


@pytest.fixture
def tickets_db(tmp_path, monkeypatch):
    db_path = tmp_path / "tickets.db"
    monkeypatch.setattr("app.tools.ticketing.TICKETS_DB_PATH", db_path)
    return db_path


@pytest.fixture(scope="session")
def tiny_store(tmp_path_factory):
    """A tiny, real FAISS index (real embeddings) over two sample policy docs."""
    from app.rag.vectorstore import load_or_build_index

    docs_dir = tmp_path_factory.mktemp("docs")
    (docs_dir / "pto.md").write_text(
        "# PTO Policy\n\n## Accrual\nEmployees accrue 1.75 days of PTO per month.\n",
        encoding="utf-8",
    )
    (docs_dir / "travel.md").write_text(
        "# Travel Policy\n\n## Booking\nBook flights through the corporate travel portal.\n",
        encoding="utf-8",
    )
    index_dir = tmp_path_factory.mktemp("index")
    return load_or_build_index(docs_dir=docs_dir, index_dir=index_dir)
