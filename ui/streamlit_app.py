"""Streamlit prototype UI for the agentic RAG policy assistant.

Shows the final answer plus an expandable step-by-step trace of which graph nodes ran,
what they decided, and how long each took - so the agentic/RAG machinery behind the answer
is visible, not just the answer itself.
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL  # noqa: E402
from app.graph import build_graph  # noqa: E402
from app.rag.ingest import load_markdown_files  # noqa: E402
from app.rag.vectorstore import load_or_build_index  # noqa: E402

st.set_page_config(page_title="Policy Assistant", page_icon="🗂️", layout="wide")


@st.cache_resource(show_spinner="Loading agent (building FAISS index, connecting to Ollama)...")
def get_graph(force_rebuild: bool = False):
    store = load_or_build_index(force_rebuild=force_rebuild)
    return build_graph(store=store), store


def render_trace(state: dict) -> None:
    trace = state.get("trace", [])
    with st.expander(f"Agent trace ({len(trace)} steps)"):
        for step in trace:
            st.markdown(f"- **{step['node']}** — {step['detail']} _(⏱ {step['duration_ms']} ms)_")

    results = state.get("results", {})
    if "policy_question" in results:
        r = results["policy_question"]
        with st.expander(f"RAG sources (confidence: {r['confidence']})"):
            for src in r.get("sources", []):
                st.markdown(f"- `{src}`")
    if "create_ticket" in results:
        t = results["create_ticket"]
        st.info(f"🎫 Support ticket **{t['ticket_id']}** opened — category `{t['category']}`, priority `{t['priority']}`")
    if state.get("escalate"):
        st.warning(f"Auto-escalated: {state.get('escalate_reason')}")


def main() -> None:
    st.title("🗂️ Internal Policy & Support Assistant")
    st.caption(
        "Agentic RAG prototype over synthetic HR/IT/Finance/Legal policy documents - "
        "ask a policy question, calculate your PTO balance, or open a support ticket."
    )

    with st.sidebar:
        st.subheader("Configuration")
        st.markdown(f"**Model:** `{OLLAMA_MODEL}`")
        st.markdown(f"**Ollama URL:** `{OLLAMA_BASE_URL}`")
        docs = load_markdown_files()
        st.markdown(f"**Policy documents indexed:** {len(docs)}")
        for name, _ in docs:
            st.markdown(f"- {name}")
        if st.button("Rebuild index"):
            get_graph.clear()
            st.session_state["_force_rebuild"] = True
            st.rerun()

        st.divider()
        st.subheader("Try asking")
        st.code("How many PTO days can I carry over?", language=None)
        st.code("I was hired 2022-01-01, took 5 days off, how many are left as of 2024-01-01?", language=None)
        st.code("Please open a ticket, I lost my laptop.", language=None)

    force_rebuild = st.session_state.pop("_force_rebuild", False)
    try:
        graph, store = get_graph(force_rebuild=force_rebuild)
    except Exception as exc:  # noqa: BLE001 - surfaced directly to the user in the UI
        st.error(
            "Could not reach the Ollama model. Make sure Ollama is running and the model "
            f"`{OLLAMA_MODEL}` has been pulled (see README). Details: {exc}"
        )
        st.stop()

    if "history" not in st.session_state:
        st.session_state["history"] = []

    for turn in st.session_state["history"]:
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            render_trace(turn["state"])

    user_query = st.chat_input("Ask about company policy, your PTO balance, or open a ticket...")
    if user_query:
        with st.chat_message("user"):
            st.write(user_query)
        with st.chat_message("assistant"):
            try:
                with st.spinner("Thinking..."):
                    state = graph.invoke({"user_query": user_query})
            except Exception as exc:  # noqa: BLE001 - surfaced directly to the user in the UI
                st.error(
                    "The agent failed to reach Ollama while answering. Make sure Ollama is "
                    f"running and the model `{OLLAMA_MODEL}` is pulled. Details: {exc}"
                )
                return
            answer = state.get("final_answer", "(no answer produced)")
            st.write(answer)
            render_trace(state)
        st.session_state["history"].append({"query": user_query, "answer": answer, "state": state})


if __name__ == "__main__":
    main()
