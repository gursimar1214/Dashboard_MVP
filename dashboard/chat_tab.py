"""
chat_tab.py

The Streamlit side of the chatbot. Renders a chat UI and POSTs each
question to the FastAPI service in chatbot/main.py (started separately
-- see chatbot/README.md). The dashboard process never talks to Ollama
or the database directly for chat questions; it only calls the API.

Follow-up questions ("and last month?", "what about india") work because
every request carries a session_id: the backend issues one on the first
message, this tab keeps it in st.session_state and sends it back each time.
"New chat" tells the backend to forget it (POST /reset) and starts fresh.

Run the backend first, from the project root (ONE worker -- the memory lives
in the server process):
    python -m uvicorn chatbot.main:app --port 8000
"""
import os

import pandas as pd
import requests
import streamlit as st

CHATBOT_API_URL = os.environ.get("CHATBOT_API_URL", "http://localhost:8000/chat")
# /reset lives next to /chat on the same server.
CHATBOT_RESET_URL = CHATBOT_API_URL.rstrip("/").removesuffix("/chat") + "/reset"
CHATBOT_FEEDBACK_URL = CHATBOT_API_URL.rstrip("/").removesuffix("/chat") + "/feedback"
# The backend may queue a message behind the previous one in the same session
# and the very first call can include loading the model, so be generous.
REQUEST_TIMEOUT_S = 180


def _render_details(sql, rows) -> None:
    """Under an answer: a table when there are several rows, and the SQL
    that produced it."""
    if rows and len(rows) > 1:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    if sql:
        with st.expander("Query used"):
            st.code(sql, language="sql")
            if rows is not None:
                st.caption(f"{len(rows)} row(s) returned")


def _render_interpretation(interpreted_as) -> None:
    """Show how a follow-up was understood, so a misreading is visible."""
    if interpreted_as:
        st.caption(f"Understood as: {interpreted_as}")


def _send_feedback(answer_id, helpful: bool, comment: str = "") -> bool:
    """Thumbs up/down (+ optional comment) -> Langfuse score on that answer's
    trace. Best effort; returns whether the server accepted it."""
    try:
        r = requests.post(
            CHATBOT_FEEDBACK_URL,
            json={"answer_id": answer_id, "helpful": helpful, "comment": comment or None},
            timeout=5,
        )
        return r.ok and r.json().get("status") == "recorded"
    except (requests.RequestException, ValueError):
        return False


def _render_feedback(answer_id) -> None:
    """Thumbs under an answer (needs a Streamlit with st.feedback). The server
    always logs it to chatbot/feedback_log.jsonl and, with Langfuse on, also
    scores the trace. Changing your mind re-sends. A thumbs-down offers a box
    for what went wrong -- that text is what makes bad answers fixable."""
    if not answer_id or not hasattr(st, "feedback"):
        return
    last = st.session_state.setdefault("chat_feedback_last", {})
    choice = st.feedback("thumbs", key=f"fb_{answer_id}")      # 1 = up, 0 = down
    if choice is not None and last.get(answer_id) != choice:
        _send_feedback(answer_id, helpful=bool(choice))
        last[answer_id] = choice
    if choice == 0:
        note = st.text_input(
            "What went wrong? (optional)", key=f"fbnote_{answer_id}",
            placeholder="e.g. wrong country filter, expected average not total",
        )
        if note and st.button("Send note", key=f"fbsend_{answer_id}"):
            if _send_feedback(answer_id, helpful=False, comment=note):
                st.caption("Thanks -- noted.")


def _new_chat() -> None:
    """Forget the conversation on the server, then locally."""
    session_id = st.session_state.get("chat_session_id")
    if session_id:
        try:
            requests.post(CHATBOT_RESET_URL, json={"session_id": session_id}, timeout=10)
        except requests.RequestException:
            pass  # server unreachable: it forgets on its own after the idle timeout
    st.session_state.chat_session_id = None
    st.session_state.chat_messages = []


def render_chatbot() -> None:
    st.caption(
        "Ask about the data behind this dashboard -- e.g. \"What's our AOV in "
        "Germany this year?\" or \"Which payment method brings in the most "
        "revenue?\" This runs fully locally through Ollama; nothing leaves "
        "this machine."
    )

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "chat_session_id" not in st.session_state:
        st.session_state.chat_session_id = None

    if st.session_state.chat_messages and st.button("New chat", help="Forget this conversation and start over"):
        _new_chat()
        st.rerun()

    for i, msg in enumerate(st.session_state.chat_messages):
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            _render_interpretation(msg.get("interpreted_as"))
            _render_details(msg.get("sql"), msg.get("rows"))
            if msg["role"] == "assistant":
                _render_feedback(msg.get("answer_id"))

    question = st.chat_input("Ask a question about the data...")
    if not question:
        return

    st.session_state.chat_messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            reply = _ask_backend(question, st.session_state.chat_session_id)
        if reply["session_id"]:
            st.session_state.chat_session_id = reply["session_id"]   # sent back on the next question
        st.markdown(reply["answer"])
        _render_interpretation(reply["interpreted_as"])
        _render_details(reply["sql"], reply["rows"])

    st.session_state.chat_messages.append({
        "role": "assistant", "content": reply["answer"], "sql": reply["sql"],
        "rows": reply["rows"], "interpreted_as": reply["interpreted_as"],
        "answer_id": reply.get("answer_id"),
    })
    st.rerun()   # redraw so the thumbs widget appears under the new answer


def _ask_backend(question: str, session_id=None) -> dict:
    """POST one question. Always returns a dict with answer, sql, rows,
    interpreted_as and session_id (None if the server was not reached, so a
    failed call never clobbers the session we already have)."""
    payload = {"message": question}
    if session_id:
        payload["session_id"] = session_id
    try:
        resp = requests.post(CHATBOT_API_URL, json=payload, timeout=REQUEST_TIMEOUT_S)
        resp.raise_for_status()
        data = resp.json()
        return {
            "answer": data["answer"],
            "sql": data.get("sql"),
            "rows": data.get("rows"),
            "interpreted_as": data.get("interpreted_as"),
            "session_id": data.get("session_id") or session_id,
            "answer_id": data.get("answer_id"),
        }
    except requests.RequestException as exc:
        return {
            "answer": (
                f"Couldn't reach the chatbot service at `{CHATBOT_API_URL}`. Is it running? "
                f"(`python -m uvicorn chatbot.main:app --port 8000` from the project root) -- {exc}"
            ),
            "sql": None, "rows": None, "interpreted_as": None, "session_id": None,
            "answer_id": None,
        }