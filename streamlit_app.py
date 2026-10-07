"""Streamlit chat frontend for the Personal Knowledge Engine.

A second, separate frontend alongside the existing vanilla HTML/JS dashboard
(app/static/) — not a replacement. Talks to the FastAPI backend purely over
HTTP, same as that dashboard does, never imports backend code directly: this
is a genuinely separate process, run with `streamlit run streamlit_app.py`
while `uvicorn app.main:app` runs separately.

Scope, stated plainly: this covers login/register, file upload, and the
chat interface with live streaming + citations + grounding — the things
that make something "look like a chatbot." It does not reimplement the
graph visualization; that stays on the HTML dashboard, since Streamlit has
no native equivalent to the D3 force-directed graph there.
"""

from __future__ import annotations

import json
import os

import requests
import streamlit as st

API_BASE_URL = os.environ.get("PKE_API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Personal Knowledge Engine", page_icon="🧠", layout="wide")


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def try_refresh_token() -> bool:
    """Trades the stored refresh token for a new access/refresh pair.
    Returns False (without raising) on any failure — an expired or
    already-used refresh token, or the API being unreachable — which the
    caller treats as "this really does need a full re-login."."""
    if "refresh_token" not in st.session_state:
        return False
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/auth/refresh", json={"refresh_token": st.session_state.refresh_token}, timeout=30
        )
    except requests.RequestException:
        return False
    if response.status_code != 200:
        return False
    data = response.json()
    st.session_state.token = data["access_token"]
    st.session_state.refresh_token = data["refresh_token"]
    return True


def api_request(method: str, path: str, timeout: int = 30, authed: bool = False, **kwargs) -> requests.Response:
    """All HTTP calls to the API funnel through here so refresh-on-401 is
    handled in exactly one place. `authed=True` attaches the current access
    token; if the API comes back 401 (token expired), it's refreshed once
    and the exact same request is retried — transparent to every caller,
    including the streaming query call (stream=True survives in **kwargs
    since the retry re-issues the request from scratch, never reusing a
    partially-read body)."""

    def _do() -> requests.Response:
        headers = {**(kwargs.get("headers") or {}), **(auth_headers() if authed else {})}
        return requests.request(method, f"{API_BASE_URL}{path}", timeout=timeout, **{**kwargs, "headers": headers})

    response = _do()
    if authed and response.status_code == 401 and try_refresh_token():
        response = _do()
    return response


def api_post(path: str, timeout: int = 30, authed: bool = False, **kwargs) -> requests.Response:
    return api_request("POST", path, timeout=timeout, authed=authed, **kwargs)


def api_get(path: str, timeout: int = 30, authed: bool = True, **kwargs) -> requests.Response:
    return api_request("GET", path, timeout=timeout, authed=authed, **kwargs)


def _handle_expired_session() -> None:
    # Reaching here means even the refresh token didn't work (expired or
    # already used) — api_request already tried a silent refresh, so this
    # really is "log in again," not a transient blip.
    for key in ("token", "refresh_token", "email", "messages"):
        st.session_state.pop(key, None)
    st.warning("Your session expired — please log in again.")
    st.rerun()


# ---------------------------------------------------------------------
# Auth: login/register in the sidebar, nothing else usable until logged in.
# ---------------------------------------------------------------------
def render_auth_sidebar() -> None:
    with st.sidebar:
        st.header("Account")

        if "token" in st.session_state:
            st.success(f"Logged in as **{st.session_state.email}**")
            if st.button("Log out", use_container_width=True):
                try:
                    api_post("/api/auth/logout", authed=True)
                except requests.RequestException:
                    pass  # logging out locally still happens even if the request itself fails
                for key in ("token", "refresh_token", "email", "messages"):
                    st.session_state.pop(key, None)
                st.rerun()
            return

        login_tab, register_tab = st.tabs(["Log in", "Register"])

        with login_tab:
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Log in", use_container_width=True):
                    _submit_auth("/api/auth/login", {"email": email, "password": password})

        with register_tab:
            with st.form("register_form"):
                full_name = st.text_input("Full name")
                email = st.text_input("Email", key="register_email")
                password = st.text_input("Password (8+ characters)", type="password", key="register_password")
                if st.form_submit_button("Register", use_container_width=True):
                    _submit_auth(
                        "/api/auth/register", {"email": email, "password": password, "full_name": full_name}
                    )


def _submit_auth(path: str, payload: dict) -> None:
    try:
        response = api_post(path, json=payload)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
        return

    if response.status_code != 200:
        detail = response.json().get("detail", response.text) if response.content else response.text
        st.error(detail)
        return

    data = response.json()
    st.session_state.token = data["access_token"]
    st.session_state.refresh_token = data["refresh_token"]
    st.session_state.email = data["email"]
    st.rerun()


# ---------------------------------------------------------------------
# Document upload — sidebar, only shown once logged in.
# ---------------------------------------------------------------------
def render_upload_sidebar() -> None:
    with st.sidebar:
        st.header("Documents")
        uploaded = st.file_uploader(
            "Upload a PDF, Markdown, HTML, DOCX, or TXT file", type=["pdf", "md", "markdown", "html", "htm", "docx", "txt"]
        )
        if uploaded is not None and st.button("Ingest", use_container_width=True):
            with st.spinner(f"Ingesting {uploaded.name}..."):
                try:
                    response = api_post(
                        "/api/ingest",
                        files={"file": (uploaded.name, uploaded.getvalue())},
                        authed=True,
                        timeout=120,  # parsing + embedding a real document can take a while
                    )
                except requests.RequestException as exc:
                    st.error(f"Could not reach the API: {exc}")
                    return

                if response.status_code == 401:
                    _handle_expired_session()
                    return

                data = response.json()
                status = data.get("status")
                if status == "ingested":
                    st.success(
                        f"{data['total_chunks']} chunks "
                        f"({data['new_chunks']} new, {data['reused_chunks']} reused embeddings)"
                    )
                elif status == "duplicate_file":
                    st.info("Already ingested.")
                elif status == "unsupported":
                    st.warning("Unsupported file type.")
                else:
                    st.error(data.get("error") or status)


# ---------------------------------------------------------------------
# Document list — "what have I actually uploaded, and how much of it."
# Streamlit reruns this whole script on every interaction, so fetching
# fresh here is enough to pick up a just-finished ingest above — no
# separate refresh hook needed.
# ---------------------------------------------------------------------
def render_documents_sidebar() -> None:
    with st.sidebar:
        st.header("Your documents")
        try:
            response = api_get("/api/documents")
        except requests.RequestException as exc:
            st.caption(f"Could not load documents: {exc}")
            return

        if response.status_code == 401:
            _handle_expired_session()
            return
        if response.status_code != 200:
            st.caption("Could not load documents.")
            return

        data = response.json()
        st.caption(f"{data['total_documents']} document(s), {data['total_chunks']} chunk(s) total")
        for doc in data["documents"]:
            st.text(f"{doc['file_name']} — {doc['chunk_count']} chunks")


# ---------------------------------------------------------------------
# Chat — streams tokens live via POST /api/query/stream, same SSE wire
# format the HTML dashboard's hand-parsed fetch() reader consumes; here
# it's requests' iter_lines() feeding a generator straight into
# st.chat_message(...).write_stream(), which is what gives the live
# token-by-token "typing" effect.
# ---------------------------------------------------------------------
def stream_answer(question: str):
    """Returns (generator_of_text_pieces, final_event_holder). The caller
    must fully consume the generator (e.g. via st.write_stream) before
    final_event_holder is populated — it's filled as a side effect of
    iterating the stream, not available up front."""
    final_event: dict = {}

    def _generate():
        response = api_post("/api/query/stream", json={"query": question}, authed=True, stream=True, timeout=120)
        if response.status_code == 401:
            final_event["type"] = "session_expired"
            return
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            event = json.loads(line[len("data:") :].strip())
            if event["type"] == "token":
                yield event["text"]
            else:
                final_event.update(event)

    return _generate(), final_event


def render_citations(citations: list[dict]) -> None:
    if not citations:
        return
    with st.expander(f"Sources ({len(citations)})"):
        for citation in citations:
            st.markdown(f"**[{citation['marker']}] {citation['file_name']}**")
            st.caption(citation["snippet"])


def render_grounding(grounding: dict | None) -> None:
    if not grounding:
        return
    coverage = grounding.get("overall_coverage", 0.0)
    unsupported = grounding.get("unsupported_sentences", [])
    caption = f"Grounding coverage: {coverage:.0%}"
    if unsupported:
        caption += f" — ⚠️ {len(unsupported)} sentence(s) not clearly supported by sources"
    st.caption(caption)


def render_chat() -> None:
    st.title("🧠 Personal Knowledge Engine")
    st.caption("Ask questions about the documents you've ingested. Answers are grounded and cited, or refused.")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message["role"] == "assistant":
                render_citations(message.get("citations", []))
                render_grounding(message.get("grounding"))

    question = st.chat_input("What does Ollama run models on?")
    if not question:
        return

    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            generator, final_event = stream_answer(question)
            full_text = st.write_stream(generator)
        except requests.RequestException as exc:
            st.error(f"Could not reach the API: {exc}")
            return

        event_type = final_event.get("type")
        if event_type == "session_expired":
            st.session_state.messages.pop()  # drop the user message we just added — it was never answered
            _handle_expired_session()
            return
        elif event_type == "refused":
            st.warning(f"No grounded answer available (reason: {final_event.get('reason')}).")
            st.session_state.messages.append({"role": "assistant", "content": "*(refused — no grounded context found)*"})
        elif event_type == "error":
            st.error(final_event.get("message", "Unknown error."))
            st.session_state.messages.append({"role": "assistant", "content": "*(error during generation)*"})
        else:
            citations = final_event.get("citations", [])
            grounding = final_event.get("grounding")
            render_citations(citations)
            render_grounding(grounding)
            st.session_state.messages.append(
                {"role": "assistant", "content": full_text, "citations": citations, "grounding": grounding}
            )


# ---------------------------------------------------------------------
render_auth_sidebar()

if "token" not in st.session_state:
    st.info("Log in or register in the sidebar to get started.")
else:
    render_upload_sidebar()
    render_documents_sidebar()
    render_chat()
