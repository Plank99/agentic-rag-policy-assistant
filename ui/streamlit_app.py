"""Streamlit prototype UI for the agentic RAG policy assistant.

Shows the final answer plus an expandable step-by-step trace of which graph nodes ran,
what they decided, and how long each took - so the agentic/RAG machinery behind the answer
is visible, not just the answer itself.
"""
from __future__ import annotations

import html
import sys
import uuid
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import OLLAMA_BASE_URL, OLLAMA_MODEL  # noqa: E402
from app.graph import build_graph  # noqa: E402
from app.llm import generate_followups, get_chat_llm  # noqa: E402
from app.rag.ingest import load_markdown_files  # noqa: E402
from app.rag.vectorstore import load_or_build_index  # noqa: E402

st.set_page_config(page_title="Policy Assistant", layout="wide")

# Streamlit's own "Deploy" button isn't relevant for this prototype; hide only that one
# toolbar element via its test id, so the rest of the native menu (incl. the light/dark
# theme switch under Settings) stays intact.
st.markdown(
    '<style>[data-testid="stAppDeployButton"] { display: none; }</style>',
    unsafe_allow_html=True,
)

CHAT_INPUT_PLACEHOLDER = "Ask about company policy, your PTO balance, or open a ticket..."
EXAMPLE_QUESTIONS = [
    "How many PTO days can I carry over?",
    "I was hired 2022-01-01, took 5 days off, how many are left as of 2024-01-01?",
    "Please open a ticket, I lost my laptop.",
]


@st.cache_resource(show_spinner="Loading agent (building FAISS index, connecting to Ollama)...")
def get_graph(force_rebuild: bool = False):
    store = load_or_build_index(force_rebuild=force_rebuild)
    return build_graph(store=store), store


@st.cache_resource(show_spinner=False)
def get_followup_llm():
    return get_chat_llm()


def _new_session() -> dict:
    return {"history": []}


def _init_sessions() -> None:
    if "sessions" not in st.session_state:
        session_id = str(uuid.uuid4())
        st.session_state["sessions"] = {session_id: _new_session()}
        st.session_state["active_session_id"] = session_id


def _session_title(session: dict) -> str:
    if not session["history"]:
        return "New chat"
    first_query = session["history"][0]["query"]
    return first_query if len(first_query) <= 40 else first_query[:40] + "…"


def render_trace(state: dict) -> None:
    trace = state.get("trace", [])
    with st.expander(f"Agent trace ({len(trace)} steps)"):
        for step in trace:
            st.markdown(f"- **{step['node']}** — {step['detail']} _({step['duration_ms']} ms)_")

    results = state.get("results", {})
    if "policy_question" in results:
        r = results["policy_question"]
        with st.expander(f"RAG sources (confidence: {r['confidence']})"):
            for src in r.get("sources", []):
                st.markdown(f"- `{src}`")
    if "create_ticket" in results:
        t = results["create_ticket"]
        st.info(
            f"Support ticket **{t['ticket_id']}** opened — category `{t['category']}`, "
            f"priority `{t['priority']}`"
        )
    if state.get("escalate"):
        st.warning(f"Auto-escalated: {state.get('escalate_reason')}")


def render_sidebar() -> None:
    with st.sidebar:
        st.markdown("### My Chats")

        for session_id, session in reversed(list(st.session_state["sessions"].items())):
            is_active = session_id == st.session_state["active_session_id"]
            title_col, delete_col = st.columns([5, 1])
            with title_col:
                if st.button(
                    _session_title(session),
                    key=f"session_{session_id}",
                    use_container_width=True,
                    type="primary" if is_active else "secondary",
                ):
                    st.session_state["active_session_id"] = session_id
                    st.rerun()
            with delete_col:
                if st.button(
                    "",
                    key=f"delete_{session_id}",
                    icon=":material/delete:",
                    help="Delete this chat",
                ):
                    del st.session_state["sessions"][session_id]
                    if not st.session_state["sessions"]:
                        new_id = str(uuid.uuid4())
                        st.session_state["sessions"][new_id] = _new_session()
                        st.session_state["active_session_id"] = new_id
                    elif st.session_state["active_session_id"] == session_id:
                        st.session_state["active_session_id"] = next(
                            iter(st.session_state["sessions"])
                        )
                    st.rerun()

        if st.button("+ New chat", key="new_chat_btn", use_container_width=True):
            session_id = str(uuid.uuid4())
            st.session_state["sessions"][session_id] = _new_session()
            st.session_state["active_session_id"] = session_id
            st.rerun()


@st.dialog("Config")
def show_config_dialog(docs: list) -> None:
    st.markdown(f"Model: `{OLLAMA_MODEL}`")
    st.markdown(f"Ollama URL: `{OLLAMA_BASE_URL}`")
    st.markdown(f"Policy documents indexed: {len(docs)}")
    for name, _ in docs:
        st.markdown(f"- {name}")
    if st.button("Rebuild index"):
        get_graph.clear()
        st.session_state["_force_rebuild"] = True
        st.rerun()


def render_question_chips(label: str, questions: list[str]) -> None:
    """Shared renderer for clickable example/follow-up question cards.

    ``st.markdown(unsafe_allow_html=True)`` sanitizes out ``<script>`` tags, so this is
    rendered through ``components.html`` (a real iframe) instead, which does execute its
    script. The script reaches across to the parent document (same origin, so
    ``window.parent`` works) to find the chat input's textarea. Clicking a card fills that
    textarea without submitting it, and the whole block hides/reappears live as the
    textarea is typed into/cleared - all client-side, no server round-trip. Python only
    controls whether this component is rendered at all for a given rerun.
    """
    if not questions:
        return
    items_html = "".join(
        f'<div class="try-asking-item" data-text="{html.escape(q)}">{html.escape(q)}</div>'
        for q in questions
    )
    components.html(
        f"""
<!doctype html>
<html>
<head>
<style>
  html, body {{ margin: 0; padding: 0; background: transparent;
                font-family: "Source Sans Pro", sans-serif; }}
  .try-asking-label {{ font-weight: 600; margin-bottom: 0.4rem; opacity: 0.85;
                        font-size: 0.9rem; }}
  .try-asking-item {{
      padding: 0.5rem 0.75rem;
      margin-bottom: 0.4rem;
      border-radius: 0.5rem;
      font-family: monospace;
      font-size: 0.85rem;
      cursor: pointer;
      transition: background-color 0.15s ease;
  }}
  body.theme-dark .try-asking-label {{ color: #E8E8E8; }}
  body.theme-dark .try-asking-item {{
      border: 1px solid rgba(255, 255, 255, 0.18);
      background-color: rgba(255, 255, 255, 0.05);
      color: #E8E8E8;
  }}
  body.theme-dark .try-asking-item:hover {{ background-color: rgba(91, 141, 239, 0.28); }}
  body.theme-light .try-asking-label {{ color: #31333F; }}
  body.theme-light .try-asking-item {{
      border: 1px solid rgba(0, 0, 0, 0.2);
      background-color: rgba(0, 0, 0, 0.035);
      color: #31333F;
  }}
  body.theme-light .try-asking-item:hover {{ background-color: rgba(91, 141, 239, 0.18); }}
</style>
</head>
<body>
<div id="try-asking-block">
  <div class="try-asking-label">{html.escape(label)}</div>
  {items_html}
</div>
<script>
(function() {{
    function getTextarea() {{
        try {{
            return window.parent.document.querySelector(
                'textarea[data-testid="stChatInputTextArea"]'
            );
        }} catch (e) {{ return null; }}
    }}

    function applyThemeClass() {{
        let dark = true;
        try {{
            const bg = window.parent.getComputedStyle(
                window.parent.document.body
            ).backgroundColor;
            const m = bg.match(/\\d+/g);
            if (m) {{
                const [r, g, b] = m.map(Number);
                dark = (0.299 * r + 0.587 * g + 0.114 * b) < 128;
            }}
        }} catch (e) {{}}
        document.body.classList.toggle('theme-dark', dark);
        document.body.classList.toggle('theme-light', !dark);
    }}
    applyThemeClass();
    // Switching Light/Dark in Streamlit's native menu is a frontend-only change (no
    // Python rerun), so this iframe wouldn't otherwise learn about it. A MutationObserver
    // on the parent app root reacts the same tick the theme's style/class actually
    // changes, instead of a polling interval that would show stale colors until the next
    // tick fires.
    try {{
        const root = window.parent.document.querySelector('[data-testid="stApp"]')
            || window.parent.document.body;
        new MutationObserver(applyThemeClass).observe(root, {{
            attributes: true, attributeFilter: ['style', 'class'], subtree: true
        }});
    }} catch (e) {{
        setInterval(applyThemeClass, 800);
    }}

    const textarea = getTextarea();
    if (!textarea) return;

    function sync() {{
        const empty = textarea.value.trim().length === 0;
        try {{ window.frameElement.style.display = empty ? '' : 'none'; }} catch (e) {{}}
    }}

    if (!textarea.dataset.suggestionsBound) {{
        textarea.addEventListener('input', sync);
        textarea.dataset.suggestionsBound = '1';
    }}
    sync();

    document.querySelectorAll('.try-asking-item').forEach(function(el) {{
        el.onclick = function() {{
            const setter = Object.getOwnPropertyDescriptor(
                window.parent.HTMLTextAreaElement.prototype, 'value'
            ).set;
            setter.call(textarea, el.getAttribute('data-text'));
            textarea.dispatchEvent(new Event('input', {{ bubbles: true }}));
            textarea.focus();
        }};
    }});
}})();
</script>
</body>
</html>
""",
        height=40 + 48 * len(questions),
    )


def render_try_asking() -> None:
    render_question_chips("Try asking", EXAMPLE_QUESTIONS)


def main() -> None:
    _init_sessions()
    docs = load_markdown_files()

    header_col, menu_col = st.columns([10, 1])
    with header_col:
        st.title("Internal Policy & Support Assistant")
        st.caption(
            "Agentic RAG prototype over synthetic HR/IT/Finance/Legal policy documents - "
            "ask a policy question, calculate your PTO balance, or open a support ticket."
        )
    with menu_col:
        if st.button("Config", use_container_width=True):
            show_config_dialog(docs)

    force_rebuild = st.session_state.pop("_force_rebuild", False)
    try:
        graph, store = get_graph(force_rebuild=force_rebuild)
    except Exception as exc:  # noqa: BLE001 - surfaced directly to the user in the UI
        st.error(
            "Could not reach the Ollama model. Make sure Ollama is running and the model "
            f"`{OLLAMA_MODEL}` has been pulled (see README). Details: {exc}"
        )
        st.stop()

    active_session = st.session_state["sessions"][st.session_state["active_session_id"]]
    active_session_id = st.session_state["active_session_id"]

    user_query = st.chat_input(CHAT_INPUT_PLACEHOLDER)

    history = active_session["history"]
    is_appending_new_turn = bool(user_query)

    for idx, turn in enumerate(history):
        with st.chat_message("user"):
            st.write(turn["query"])
        with st.chat_message("assistant"):
            st.write(turn["answer"])
            render_trace(turn["state"])
            st.feedback("thumbs", key=f"feedback_{active_session_id}_{idx}")
            if idx == len(history) - 1 and not is_appending_new_turn:
                render_question_chips("You might also ask", turn.get("followups", []))

    if not history and not user_query:
        render_try_asking()

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
            with st.spinner("Generating follow-up suggestions..."):
                followups = generate_followups(get_followup_llm(), user_query, answer)
            st.feedback("thumbs", key=f"feedback_{active_session_id}_{len(history)}")
            render_question_chips("You might also ask", followups)
        active_session["history"].append(
            {"query": user_query, "answer": answer, "state": state, "followups": followups}
        )

    render_sidebar()


if __name__ == "__main__":
    main()
