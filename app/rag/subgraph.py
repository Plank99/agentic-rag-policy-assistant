"""Corrective-RAG subgraph: retrieve -> grade -> (rewrite -> retrieve)* -> generate.

This is the dedicated, modular RAG algraph referenced in the architecture doc. It is
compiled independently of the main agent graph and invoked from the `rag_answer` node as a
single callable, so it could equally be dropped into a different agent unchanged.

Self-correction loop: if the grader decides the retrieved chunks are not relevant, the query
is rewritten (e.g. to resolve an ambiguous pronoun or expand an acronym) and retrieval is
retried, up to `max_retries` times, before falling back to a low-confidence answer.
"""
from __future__ import annotations

import time

from langgraph.graph import END, StateGraph

from app.config import RAG_MAX_REWRITES, RAG_TOP_K
from app.llm import structured_json_call
from app.rag.vectorstore import similarity_search
from app.state import RagState
from app.tracing import trace_step as _trace


def build_rag_subgraph(llm, store, top_k: int = RAG_TOP_K, max_rewrites: int = RAG_MAX_REWRITES):
    def retrieve(state: RagState) -> dict:
        start = time.time()
        docs = similarity_search(store, state["query"], k=top_k)
        update = {"docs": docs}
        update.update(_trace(state, "retrieve", f"{len(docs)} chunks for: {state['query']!r}", start))
        return update

    def grade_relevance(state: RagState) -> dict:
        start = time.time()
        snippets = "\n---\n".join(d["content"] for d in state["docs"])
        prompt = (
            "You are grading whether retrieved document snippets are sufficient to answer a "
            "user question. Respond with ONLY a JSON object: {\"relevant\": true or false}.\n\n"
            f"Question: {state['query']}\n\nSnippets:\n{snippets}\n"
        )
        result = structured_json_call(llm, prompt, default={"relevant": bool(state["docs"])})
        relevant = bool(result.get("relevant", False))
        update = {"relevant": relevant}
        update.update(_trace(state, "grade_relevance", f"relevant={relevant}", start))
        return update

    def rewrite_query(state: RagState) -> dict:
        start = time.time()
        prompt = (
            "Rewrite the following question to be clearer and more specific for a vector "
            "search over internal company policy documents (resolve pronouns, expand "
            "abbreviations, add likely policy keywords). Respond with ONLY a JSON object: "
            '{"rewritten_query": "..."}.\n\n'
            f"Original question: {state['query']}\n"
        )
        result = structured_json_call(llm, prompt, default={"rewritten_query": state["query"]})
        new_query = result.get("rewritten_query") or state["query"]
        update = {
            "query": new_query,
            "retries": state.get("retries", 0) + 1,
        }
        update.update(_trace(state, "rewrite_query", f"-> {new_query!r}", start))
        return update

    def generate_answer(state: RagState) -> dict:
        start = time.time()
        if state.get("relevant"):
            snippets = "\n---\n".join(
                f"[source: {d['source']}]\n{d['content']}" for d in state["docs"]
            )
            prompt = (
                "Answer the user's question using ONLY the information in the snippets below. "
                "Cite the source filename(s) you used in parentheses. If the snippets truly do "
                "not contain the answer, say so plainly.\n\n"
                f"Question: {state['original_query']}\n\nSnippets:\n{snippets}\n"
            )
            response = llm.invoke(prompt)
            answer = response.content if hasattr(response, "content") else str(response)
            confidence = "high"
            sources = sorted({d["source"] for d in state["docs"]})
        else:
            answer = (
                "I could not find a confident answer to this in the internal policy "
                "documents after rewriting the query. This has been escalated to a support "
                "ticket so a human can follow up."
            )
            confidence = "low"
            sources = sorted({d["source"] for d in state.get("docs", [])})

        update = {"answer": answer, "confidence": confidence, "sources": sources}
        update.update(_trace(state, "generate_answer", f"confidence={confidence}", start))
        return update

    def route_after_grade(state: RagState) -> str:
        if state.get("relevant"):
            return "generate_answer"
        if state.get("retries", 0) < max_rewrites:
            return "rewrite_query"
        return "generate_answer"

    graph = StateGraph(RagState)
    graph.add_node("retrieve", retrieve)
    graph.add_node("grade_relevance", grade_relevance)
    graph.add_node("rewrite_query", rewrite_query)
    graph.add_node("generate_answer", generate_answer)

    graph.set_entry_point("retrieve")
    graph.add_edge("retrieve", "grade_relevance")
    graph.add_conditional_edges(
        "grade_relevance",
        route_after_grade,
        {"generate_answer": "generate_answer", "rewrite_query": "rewrite_query"},
    )
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("generate_answer", END)

    return graph.compile()


def run_rag(compiled_subgraph, query: str) -> dict:
    """Convenience entry point used by the main graph's rag_answer node."""
    initial: RagState = {
        "query": query,
        "original_query": query,
        "retries": 0,
        "max_retries": RAG_MAX_REWRITES,
        "trace": [],
    }
    return compiled_subgraph.invoke(initial)
