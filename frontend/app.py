"""
RAG Chatbot — Streamlit Frontend
==================================
A visually engaging chat interface for the RAG chatbot.

Features:
  - Chat bubbles with user/assistant styling
  - Document upload in the sidebar with progress feedback
  - Source transparency — shows which document chunks Gemini used
  - Session management — each browser tab gets a unique session ID
  - Chat history persisted via Firestore (survives page refresh)

Run with:
  streamlit run frontend/app.py
"""

import os
import uuid
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")

# ─────────────────────────────────────────────
# Page configuration — must be the FIRST Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────
# Custom CSS — injected into the page HTML
# Streamlit allows arbitrary HTML/CSS via st.markdown(unsafe_allow_html=True)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* ── Page background ── */
    .stApp {
        background-color: #0f1117;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d2e 0%, #0f1117 100%);
        border-right: 1px solid #2d2f3e;
    }

    /* ── App header banner ── */
    .app-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        margin-bottom: 1.5rem;
        text-align: center;
    }
    .app-header h1 {
        color: white;
        font-size: 1.8rem;
        margin: 0;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .app-header p {
        color: rgba(255,255,255,0.8);
        margin: 0.3rem 0 0 0;
        font-size: 0.9rem;
    }

    /* ── Chat message bubbles ── */
    .user-bubble {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 0.9rem 1.2rem;
        border-radius: 18px 18px 4px 18px;
        margin: 0.5rem 0;
        max-width: 80%;
        margin-left: auto;
        box-shadow: 0 2px 8px rgba(102,126,234,0.3);
        line-height: 1.5;
    }
    .assistant-bubble {
        background: #1e2130;
        color: #e0e0e0;
        padding: 0.9rem 1.2rem;
        border-radius: 18px 18px 18px 4px;
        margin: 0.5rem 0;
        max-width: 80%;
        border: 1px solid #2d2f3e;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        line-height: 1.5;
    }

    /* ── Source card ── */
    .source-card {
        background: #12141f;
        border: 1px solid #2d2f3e;
        border-left: 3px solid #667eea;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        margin: 0.3rem 0;
        font-size: 0.82rem;
        color: #9ca3af;
    }
    .source-badge {
        display: inline-block;
        background: #667eea22;
        color: #667eea;
        border: 1px solid #667eea44;
        padding: 0.15rem 0.5rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        margin-bottom: 0.4rem;
    }

    /* ── Status pills ── */
    .status-ok {
        background: #065f4622;
        color: #34d399;
        border: 1px solid #34d39944;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
    }
    .status-error {
        background: #7f1d1d22;
        color: #f87171;
        border: 1px solid #f8717144;
        padding: 0.2rem 0.7rem;
        border-radius: 20px;
        font-size: 0.78rem;
    }

    /* ── Sidebar section labels ── */
    .sidebar-label {
        color: #6b7280;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        margin: 1.2rem 0 0.4rem 0;
    }

    /* ── Session info box ── */
    .session-box {
        background: #1e2130;
        border: 1px solid #2d2f3e;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        font-size: 0.82rem;
        color: #9ca3af;
    }
    .session-box span {
        color: #667eea;
        font-family: monospace;
        font-size: 0.78rem;
    }

    /* ── Streaming cursor ── */
    @keyframes blink { 0%, 100% { opacity: 1; } 50% { opacity: 0; } }
    .cursor { animation: blink 0.7s infinite; color: #667eea; }

    /* ── Divider ── */
    hr { border-color: #2d2f3e; }

    /* ── Streamlit overrides ── */
    .stButton > button {
        border-radius: 8px;
        font-weight: 600;
    }
    [data-testid="stFileUploader"] {
        background: #1e2130;
        border: 1px dashed #2d2f3e;
        border-radius: 10px;
    }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# Session state initialisation
# st.session_state persists values during a browser session but resets on
# page refresh. To survive refreshes, we store session_id in the URL as a
# query parameter (e.g. ?session_id=session-abc123).
#
# Flow:
#   First visit  → generate new ID → write to URL and session_state
#   On refresh   → read ID from URL → restore session_state from it
#   New Session  → generate new ID → update URL and session_state
# ─────────────────────────────────────────────

# Read session_id from URL query params (survives page refresh)
params = st.query_params
url_session_id = params.get("session_id", None)

if "session_id" not in st.session_state:
    if url_session_id:
        # Returning visitor — restore the session from the URL
        st.session_state.session_id = url_session_id
    else:
        # First visit — generate a new session ID and write it to the URL
        st.session_state.session_id = f"session-{uuid.uuid4().hex[:8]}"
        st.query_params["session_id"] = st.session_state.session_id

if "messages" not in st.session_state:
    st.session_state.messages = []  # list of {"role": ..., "content": ..., "sources": ...}

if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = []  # list of successfully uploaded filenames

if "pending_question" not in st.session_state:
    # Holds the user's question while we wait for the API response.
    # None means no request in flight.
    st.session_state.pending_question = None


# ─────────────────────────────────────────────
# Helper functions
# ─────────────────────────────────────────────
def check_backend_health() -> bool:
    """Ping the FastAPI backend to check if it's running."""
    try:
        r = requests.get(f"{BACKEND_URL}/health", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def upload_document(file) -> dict:
    """Send a file to POST /upload and return the response."""
    response = requests.post(
        f"{BACKEND_URL}/upload",
        files={"file": (file.name, file.getvalue(), file.type)},
        timeout=120,
    )
    response.raise_for_status()
    return response.json()


def stream_message(session_id: str, message: str):
    """
    Stream a chat message from POST /chat/stream.

    Yields text tokens one by one as Gemini produces them, then yields the
    final __SOURCES__ chunk so the caller can parse source metadata.

    Uses requests with stream=True + iter_content(chunk_size=1) to read
    the response byte by byte as it arrives.
    """
    with requests.post(
        f"{BACKEND_URL}/chat/stream",
        json={"session_id": session_id, "message": message},
        stream=True,
        timeout=120,
    ) as response:
        response.raise_for_status()
        buffer = ""
        for raw_chunk in response.iter_content(chunk_size=None, decode_unicode=True):
            if raw_chunk:
                buffer += raw_chunk
                # Check if the buffer contains the sources marker
                if "__SOURCES__" in buffer:
                    # Split at the marker: everything before is text, after is JSON
                    text_part, sources_part = buffer.split("__SOURCES__", 1)
                    if text_part:
                        yield ("text", text_part)
                    yield ("sources", sources_part)
                    buffer = ""
                    break
                else:
                    yield ("text", raw_chunk)
                    buffer = ""


def load_history(session_id: str) -> list:
    """Load chat history from GET /history."""
    response = requests.get(
        f"{BACKEND_URL}/history",
        params={"session_id": session_id},
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("messages", [])


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 RAG Chatbot")
    st.markdown("*Powered by Gemini + Vertex AI*")

    # Backend status
    is_healthy = check_backend_health()
    if is_healthy:
        st.markdown('<span class="status-ok">● Backend connected</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="status-error">● Backend offline</span>', unsafe_allow_html=True)
        st.warning("Start the backend: `uvicorn backend.main:app --reload`")

    st.markdown("---")

    # ── Document Upload ──
    st.markdown('<p class="sidebar-label">📄 Upload Document</p>', unsafe_allow_html=True)
    st.caption("Supports PDF and TXT files")

    uploaded_file = st.file_uploader(
        label="Choose a file",
        type=["pdf", "txt"],
        label_visibility="collapsed",
    )

    if uploaded_file:
        if st.button("⚡ Process Document", use_container_width=True, type="primary"):
            with st.spinner(f"Processing '{uploaded_file.name}'..."):
                try:
                    result = upload_document(uploaded_file)
                    st.success(f"✓ {result['chunks_stored']} chunks stored")
                    if uploaded_file.name not in st.session_state.uploaded_docs:
                        st.session_state.uploaded_docs.append(uploaded_file.name)
                except requests.exceptions.ConnectionError:
                    st.error("Cannot reach backend. Is it running?")
                except Exception as e:
                    st.error(f"Upload failed: {str(e)}")

    # Show uploaded docs
    if st.session_state.uploaded_docs:
        st.markdown('<p class="sidebar-label">📚 Processed Documents</p>', unsafe_allow_html=True)
        for doc in st.session_state.uploaded_docs:
            st.markdown(f"✓ `{doc}`")

    st.markdown("---")

    # ── Session Info ──
    st.markdown('<p class="sidebar-label">💬 Session</p>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="session-box">
        ID: <span>{st.session_state.session_id}</span><br>
        Messages: <span>{len(st.session_state.messages)}</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.button("🔄 New Session", use_container_width=True):
            new_id = f"session-{uuid.uuid4().hex[:8]}"
            st.session_state.session_id = new_id
            st.session_state.messages = []
            # Update URL so the new session persists on refresh too
            st.query_params["session_id"] = new_id
            st.rerun()

    st.markdown("---")
    st.caption("Phase 5 — Local Development")
    st.caption(f"Backend: `{BACKEND_URL}`")


# ─────────────────────────────────────────────
# MAIN AREA
# ─────────────────────────────────────────────

# Header banner
st.markdown("""
<div class="app-header">
    <h1>🤖 RAG Chatbot</h1>
    <p>Ask questions about your uploaded documents — answers grounded in your content</p>
</div>
""", unsafe_allow_html=True)

# ── Load history from Firestore on first load ──
# If page is refreshed, reload previous messages from Firestore
if not st.session_state.messages:
    try:
        history = load_history(st.session_state.session_id)
        for msg in history:
            st.session_state.messages.append({
                "role": msg["role"],
                "content": msg["content"],
                "sources": [],  # sources not stored in Firestore (only answer is)
            })
    except Exception:
        pass  # silently ignore if backend is down or no history yet

# ── Display chat messages ──
if not st.session_state.messages:
    # Empty state — show a helpful prompt
    st.markdown("""
    <div style="text-align:center; padding: 3rem; color: #6b7280;">
        <div style="font-size: 3rem; margin-bottom: 1rem;">📄</div>
        <div style="font-size: 1.1rem; font-weight: 600; color: #9ca3af;">
            Start by uploading a document
        </div>
        <div style="font-size: 0.9rem; margin-top: 0.5rem;">
            Upload a PDF or TXT in the sidebar, then ask a question below
        </div>
    </div>
    """, unsafe_allow_html=True)
else:
    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">👤 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f'<div class="assistant-bubble">🤖 {msg["content"]}</div>',
                unsafe_allow_html=True,
            )

            # Show source chunks if available
            sources = msg.get("sources", [])
            if sources:
                with st.expander(f"📎 {len(sources)} source chunk(s) used", expanded=False):
                    for i, src in enumerate(sources):
                        st.markdown(f"""
                        <div class="source-card">
                            <span class="source-badge">
                                #{i+1} · {src['source']} · similarity: {1 - src['distance']:.2%}
                            </span><br>
                            {src['text'][:300]}{'...' if len(src['text']) > 300 else ''}
                        </div>
                        """, unsafe_allow_html=True)

# ── Step A: Stream API response if a question is pending ──
# This runs AFTER the rerun that showed the user bubble.
# We stream tokens from /chat/stream and update an st.empty() container
# in real time — the text grows word by word instead of appearing all at once.
if st.session_state.pending_question:
    question = st.session_state.pending_question

    # Create a placeholder inside a styled bubble so the streaming text
    # appears in the same dark bubble as regular assistant messages.
    stream_container = st.empty()
    # Show immediately — covers the embedding + TTFT delay before first token
    stream_container.markdown(
        '<div class="assistant-bubble">🤖 <em style="color:#6b7280">Thinking...</em><span class="cursor">▌</span></div>',
        unsafe_allow_html=True,
    )
    accumulated = ""
    sources = []

    try:
        for event_type, payload in stream_message(st.session_state.session_id, question):
            if event_type == "text":
                accumulated += payload
                # Re-render the bubble with a blinking cursor while streaming
                stream_container.markdown(
                    f'<div class="assistant-bubble">🤖 {accumulated}<span class="cursor">▌</span></div>',
                    unsafe_allow_html=True,
                )
            elif event_type == "sources":
                # Parse the JSON array of source chunks sent at end of stream
                try:
                    import json
                    sources = json.loads(payload)
                except Exception:
                    sources = []

        # Replace the streaming bubble with the final clean version (no cursor)
        stream_container.markdown(
            f'<div class="assistant-bubble">🤖 {accumulated}</div>',
            unsafe_allow_html=True,
        )

        st.session_state.messages.append({
            "role": "assistant",
            "content": accumulated,
            "sources": sources,
        })

    except requests.exceptions.ConnectionError:
        stream_container.empty()
        st.session_state.messages.append({
            "role": "assistant",
            "content": "⚠️ Cannot reach the backend. Is `uvicorn` running?",
            "sources": [],
        })
    except Exception as e:
        stream_container.empty()
        st.session_state.messages.append({
            "role": "assistant",
            "content": f"⚠️ Error: {str(e)}",
            "sources": [],
        })
    finally:
        st.session_state.pending_question = None

    st.rerun()  # rerun — persists the final bubble and shows sources expander

# ── Step B: Accept new user input ──
# st.chat_input stays pinned to the bottom of the page.
# When the user hits Enter:
#   1. Add their message bubble to state immediately
#   2. Store the question as pending
#   3. rerun() #1 — shows the user bubble instantly, then Step A fires
if question := st.chat_input("Ask a question about your documents..."):
    if not is_healthy:
        st.error("Backend is offline. Please start the FastAPI server first.")
    else:
        st.session_state.messages.append({
            "role": "user",
            "content": question,
            "sources": [],
        })
        st.session_state.pending_question = question
        st.rerun()  # rerun #1 — user bubble appears immediately