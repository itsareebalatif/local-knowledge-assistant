from __future__ import annotations

import json
import os
import time

import requests
import streamlit as st

API_BASE_URL = os.environ.get("PKE_API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="Personal Knowledge Engine", page_icon="🧠", layout="wide")


def auth_headers() -> dict:
    return {"Authorization": f"Bearer {st.session_state.token}"}


def try_refresh_token() -> bool:
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
    # The silent refresh already failed, so a full re-login is needed.
    for key in ("token", "refresh_token", "email", "messages", "conversation_id"):
        st.session_state.pop(key, None)
    st.warning("Your session expired — please log in again.")
    st.rerun()


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
                for key in ("token", "refresh_token", "email", "messages", "conversation_id"):
                    st.session_state.pop(key, None)
                st.rerun()
            return

        login_tab, register_tab = st.tabs(["Log in", "Register"])

        with login_tab:
            # Can't combine this with the form below (ruff SIM117) — the
            # forgot-password expander is a sibling block inside login_tab
            # that comes after the form closes, not inside it.
            with st.form("login_form"):
                email = st.text_input("Email")
                password = st.text_input("Password", type="password")
                if st.form_submit_button("Log in", use_container_width=True):
                    _submit_auth("/api/auth/login", {"email": email, "password": password})

            with st.expander("Forgot password?"):
                st.caption("Enter your account email and we'll send you a link to reset it.")
                with st.form("forgot_password_form"):
                    reset_email = st.text_input("Email", key="forgot_password_email")
                    if st.form_submit_button("Send reset link", use_container_width=True):
                        _submit_forgot_password(reset_email)

        with register_tab, st.form("register_form"):
            full_name = st.text_input("Full name")
            email = st.text_input("Email", key="register_email")
            password = st.text_input("Password (8+ characters)", type="password", key="register_password")
            if st.form_submit_button("Register", use_container_width=True):
                _submit_auth(
                    "/api/auth/register", {"email": email, "password": password, "full_name": full_name}
                )


def _submit_forgot_password(email: str) -> None:
    if not email:
        st.warning("Enter your email first.")
        return
    try:
        response = api_post("/api/auth/forgot-password", json={"email": email})
    except requests.RequestException as exc:
        st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
        return
    # Always 200 regardless of whether the account exists, to avoid leaking that info.
    if response.status_code == 200:
        st.success(response.json().get("message", "If that email has an account, a reset link is on its way."))
    else:
        detail = response.json().get("detail", response.text) if response.content else response.text
        st.error(detail)


def render_reset_password_page(token: str) -> None:
    _left, center, _right = st.columns([1, 2, 1])
    with center:
        st.markdown(
            "<div style='text-align: center; margin-top: 2rem;'>"
            "<div style='font-size: 2rem;'>🧠</div>"
            "<h2 style='margin-bottom: 0;'>Reset your password</h2>"
            "<p style='color: #6b7280;'>Choose a new password for your Personal Knowledge Engine account.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

        with st.form("reset_password_form"):
            new_password = st.text_input("New password (8+ characters)", type="password")
            confirm_password = st.text_input("Confirm new password", type="password")
            submitted = st.form_submit_button("Reset password", use_container_width=True)

        if not submitted:
            return
        if new_password != confirm_password:
            st.error("Passwords don't match.")
            return
        if len(new_password) < 8:
            st.error("Password must be at least 8 characters.")
            return

        try:
            response = api_post("/api/auth/reset-password", json={"token": token, "new_password": new_password})
        except requests.RequestException as exc:
            st.error(f"Could not reach the API at {API_BASE_URL}: {exc}")
            return

        if response.status_code == 200:
            st.success("Password reset. You can now log in with your new password.")
            st.query_params.clear()
            if st.button("Go to login", use_container_width=True):
                st.rerun()
        else:
            detail = response.json().get("detail", response.text) if response.content else response.text
            st.error(f"{detail} — the link may have expired or already been used. Request a new one from the login page.")


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


def render_upload_sidebar() -> None:
    with st.sidebar:
        st.header("Documents")
        # accept_multiple_files=True — without it, st.file_uploader silently
        # keeps only the last file selected/dropped and drops the rest, no
        # error, no warning. Confirmed live: that's exactly what looked like
        # "only the most recent upload went through."
        uploaded_files = st.file_uploader(
            "Upload PDF, Markdown, HTML, DOCX, or TXT files",
            type=["pdf", "md", "markdown", "html", "htm", "docx", "txt"],
            accept_multiple_files=True,
        )
        if uploaded_files and st.button("Ingest", use_container_width=True):
            # One request per file — the API only ever accepts one file per
            # call (see app/api/ingest_routes.py) — same pattern the
            # dashboard (app/static/app.js) already uses for this.
            for uploaded in uploaded_files:
                with st.spinner(f"Ingesting {uploaded.name}..."):
                    try:
                        response = api_post(
                            "/api/ingest",
                            files={"file": (uploaded.name, uploaded.getvalue())},
                            authed=True,
                            timeout=120,  # parsing + embedding a real document can take a while
                        )
                    except requests.RequestException as exc:
                        st.error(f"{uploaded.name}: could not reach the API: {exc}")
                        continue

                    if response.status_code == 401:
                        _handle_expired_session()
                        return

                    data = response.json()
                    status = data.get("status")
                    if status == "ingested":
                        st.success(
                            f"{uploaded.name}: {data['total_chunks']} chunks "
                            f"({data['new_chunks']} new, {data['reused_chunks']} reused embeddings)"
                        )
                    elif status == "duplicate_file":
                        st.info(f"{uploaded.name}: already ingested.")
                    elif status == "unsupported":
                        st.warning(f"{uploaded.name}: unsupported file type.")
                    else:
                        st.error(f"{uploaded.name}: {data.get('error') or status}")


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
            col_name, col_delete = st.columns([5, 1])
            col_name.text(f"{doc['file_name']} — {doc['chunk_count']} chunks")
            # key=doc_id keeps each row's button/confirm state independent.
            if col_delete.button("🗑️", key=f"delete_{doc['doc_id']}", help="Delete this document"):
                st.session_state[f"confirm_delete_{doc['doc_id']}"] = True

            if st.session_state.get(f"confirm_delete_{doc['doc_id']}"):
                st.warning(f"Delete **{doc['file_name']}** and all its chunks/embeddings? This can't be undone.")
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button("Delete", key=f"confirm_{doc['doc_id']}", use_container_width=True):
                    _delete_document(doc["doc_id"])
                if cancel_col.button("Cancel", key=f"cancel_{doc['doc_id']}", use_container_width=True):
                    del st.session_state[f"confirm_delete_{doc['doc_id']}"]
                    st.rerun()


def _delete_document(doc_id: int) -> None:
    try:
        response = api_request("DELETE", f"/api/documents/{doc_id}", authed=True)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
        return

    if response.status_code == 401:
        _handle_expired_session()
        return

    st.session_state.pop(f"confirm_delete_{doc_id}", None)
    if response.status_code == 200:
        st.success("Document deleted.")
        st.rerun()
    else:
        detail = response.json().get("detail", response.text) if response.content else response.text
        st.error(detail)


def stream_answer(question: str):
    # final_event is populated as a side effect of consuming the generator,
    # so it's only valid once the caller has fully iterated it.
    final_event: dict = {}

    def _generate():
        payload = {"query": question, "conversation_id": st.session_state.get("conversation_id")}
        response = api_post("/api/query/stream", json=payload, authed=True, stream=True, timeout=120)
        if response.status_code == 401:
            final_event["type"] = "session_expired"
            return
        response.raise_for_status()
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data:"):
                continue
            event = json.loads(line[len("data:") :].strip())
            if event["type"] == "token":
                # Purely cosmetic pacing; tokens actually arrive much faster.
                time.sleep(0.018)
                yield event["text"]
            else:
                final_event.update(event)

    return _generate(), final_event


def _refusal_message(reason: str | None) -> str:
    if reason == "no_candidates":
        return "I couldn't find anything in your documents related to that. Try rephrasing, or upload a document that covers this topic."
    if reason == "low_keyword_coverage":
        return "I don't have enough information in your documents to answer that confidently. Try rephrasing using wording closer to what's actually in the document, or ask about something more specific."
    return "I don't have enough grounded information in your documents to answer that."


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
    coverage = grounding.get("overall_coverage")
    if coverage is None:
        st.caption("No factual claims to verify.")
        return
    unsupported = grounding.get("unsupported_sentences", [])
    caption = f"Grounding coverage: {coverage:.0%}"
    if unsupported:
        caption += f" — ⚠️ {len(unsupported)} sentence(s) not clearly supported by sources"
    st.caption(caption)


def render_history_sidebar() -> None:
    with st.sidebar:
        st.header("Chat history")

        if st.button("➕ New chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.conversation_id = None
            st.rerun()

        try:
            response = api_get("/api/conversations")
        except requests.RequestException as exc:
            st.caption(f"Could not load history: {exc}")
            return
        if response.status_code == 401:
            _handle_expired_session()
            return
        if response.status_code != 200:
            st.caption("Could not load history.")
            return

        conversations = response.json()["conversations"]
        if not conversations:
            st.caption("No chats yet.")

        active_id = st.session_state.get("conversation_id")
        for conv in conversations:
            col_title, col_delete = st.columns([5, 1])
            label = ("➡️ " if conv["conversation_id"] == active_id else "") + conv["title"]
            if col_title.button(label, key=f"open_conv_{conv['conversation_id']}", use_container_width=True):
                _load_conversation(conv["conversation_id"])
            if col_delete.button("🗑️", key=f"delete_conv_{conv['conversation_id']}", help="Delete this chat"):
                st.session_state[f"confirm_delete_conv_{conv['conversation_id']}"] = True

            if st.session_state.get(f"confirm_delete_conv_{conv['conversation_id']}"):
                st.warning(f"Delete **{conv['title']}**? This can't be undone.")
                confirm_col, cancel_col = st.columns(2)
                if confirm_col.button(
                    "Delete", key=f"confirm_delete_conv_btn_{conv['conversation_id']}", use_container_width=True
                ):
                    _delete_conversation(conv["conversation_id"])
                if cancel_col.button(
                    "Cancel", key=f"cancel_delete_conv_{conv['conversation_id']}", use_container_width=True
                ):
                    del st.session_state[f"confirm_delete_conv_{conv['conversation_id']}"]
                    st.rerun()


def _load_conversation(conversation_id: int) -> None:
    try:
        response = api_get(f"/api/conversations/{conversation_id}")
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
        return
    if response.status_code == 401:
        _handle_expired_session()
        return
    if response.status_code != 200:
        st.error("Could not load that chat.")
        return

    data = response.json()
    st.session_state.conversation_id = data["conversation_id"]
    st.session_state.messages = [
        {
            "role": m["role"],
            "content": m["content"],
            "citations": m.get("citations") or [],
            "grounding": m.get("grounding"),
        }
        for m in data["messages"]
    ]
    st.rerun()


def _delete_conversation(conversation_id: int) -> None:
    try:
        response = api_request("DELETE", f"/api/conversations/{conversation_id}", authed=True)
    except requests.RequestException as exc:
        st.error(f"Could not reach the API: {exc}")
        return
    if response.status_code == 401:
        _handle_expired_session()
        return

    st.session_state.pop(f"confirm_delete_conv_{conversation_id}", None)
    if response.status_code == 200:
        if st.session_state.get("conversation_id") == conversation_id:
            st.session_state.conversation_id = None
            st.session_state.messages = []
        st.rerun()
    else:
        detail = response.json().get("detail", response.text) if response.content else response.text
        st.error(detail)


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

        st.session_state.conversation_id = final_event.get("conversation_id")

        if event_type == "refused":
            message = _refusal_message(final_event.get("reason"))
            st.markdown(message)
            st.session_state.messages.append({"role": "assistant", "content": message})
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


_reset_token = st.query_params.get("token")
if _reset_token:
    render_reset_password_page(_reset_token)
else:
    render_auth_sidebar()

    if "token" not in st.session_state:
        st.info("Log in or register in the sidebar to get started.")
    else:
        render_history_sidebar()
        render_upload_sidebar()
        render_documents_sidebar()
        render_chat()
