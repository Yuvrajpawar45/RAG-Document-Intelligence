"""
DocMind - RAG Document Intelligence
Streamlit Frontend - Professional Warm UI
Run: streamlit run app.py
"""

import os
import streamlit as st
import requests
import json
from dotenv import load_dotenv

load_dotenv()
API = os.getenv("DOCMIND_API_URL", "http://localhost:8001").rstrip("/")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocMind - RAG Document Intelligence",
    page_icon="📖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600&family=Instrument+Sans:wght@400;500;600&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Instrument Sans', sans-serif;
}

.stApp {
    background-color: #faf7f2;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 1rem; }

/* ── Top header bar ── */
.docmind-header {
    background: #ffffff;
    border: 1px solid #ede5d8;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    margin-bottom: 1.25rem;
    display: flex;
    align-items: center;
    justify-content: space-between;
    box-shadow: 0 1px 4px rgba(28,22,18,0.07);
}
.docmind-logo {
    font-family: 'Fraunces', serif;
    font-size: 1.4rem;
    font-weight: 600;
    color: #1c1612;
    letter-spacing: -0.02em;
}
.docmind-logo span { color: #c8701a; }
.docmind-tagline { font-size: 0.78rem; color: #9c8878; margin-top: 0.1rem; }

/* ── Stat badges ── */
.stat-row { display: flex; gap: 0.6rem; flex-wrap: wrap; }
.stat-badge {
    background: #f5e6d0;
    border: 1px solid rgba(200,112,26,0.2);
    color: #a85c10;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
}
.status-online {
    background: #e8f5e8;
    border: 1px solid rgba(45,122,58,0.2);
    color: #2d7a3a;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
}
.status-offline {
    background: #fdecea;
    border: 1px solid rgba(192,57,43,0.2);
    color: #c0392b;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
}

/* ── Section labels ── */
.section-label {
    font-size: 0.65rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9c8878;
    margin-bottom: 0.5rem;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid #ede5d8;
}

/* ── Chat messages ── */
.chat-user {
    background: linear-gradient(135deg, #c8701a, #a85c10);
    color: white;
    border-radius: 14px 14px 4px 14px;
    padding: 0.85rem 1.1rem;
    margin: 0.4rem 0;
    margin-left: 15%;
    font-size: 0.875rem;
    line-height: 1.6;
    box-shadow: 0 2px 8px rgba(200,112,26,0.2);
}
.chat-label-user {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9c8878;
    text-align: right;
    margin-right: 0.2rem;
    margin-bottom: 0.2rem;
}
.chat-bot {
    background: #ffffff;
    border: 1px solid #ede5d8;
    border-radius: 14px 14px 14px 4px;
    padding: 0.85rem 1.1rem;
    margin: 0.4rem 0;
    margin-right: 10%;
    font-size: 0.875rem;
    line-height: 1.65;
    color: #1c1612;
    box-shadow: 0 1px 4px rgba(28,22,18,0.06);
}
.chat-label-bot {
    font-size: 0.62rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #9c8878;
    margin-bottom: 0.2rem;
}
.chat-meta {
    margin-top: 0.5rem;
    padding-top: 0.4rem;
    border-top: 1px solid #f4efe6;
    font-size: 0.68rem;
    color: #9c8878;
}
.source-pill {
    display: inline-block;
    background: #f5e6d0;
    border: 1px solid rgba(200,112,26,0.2);
    color: #a85c10;
    font-size: 0.65rem;
    font-weight: 500;
    padding: 1px 7px;
    border-radius: 4px;
    margin: 1px 2px;
}

/* ── Empty state ── */
.empty-state {
    text-align: center;
    padding: 3rem 2rem;
    color: #9c8878;
}
.empty-icon { font-size: 2.5rem; margin-bottom: 0.75rem; }
.empty-title {
    font-family: 'Fraunces', serif;
    font-size: 1.15rem;
    color: #6b5a47;
    margin-bottom: 0.4rem;
}
.empty-sub { font-size: 0.82rem; line-height: 1.6; max-width: 340px; margin: 0 auto; }

/* ── Example chips ── */
.chips-row { display: flex; flex-wrap: wrap; gap: 0.4rem; justify-content: center; margin-top: 1rem; }
.chip-item {
    background: #ffffff;
    border: 1px solid #ede5d8;
    color: #6b5a47;
    font-size: 0.74rem;
    padding: 0.4rem 0.9rem;
    border-radius: 20px;
    cursor: pointer;
}

/* ── Panel card ── */
.panel-card {
    background: #ffffff;
    border: 1px solid #ede5d8;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 0.75rem;
    box-shadow: 0 1px 3px rgba(28,22,18,0.05);
}

/* ── File item ── */
.file-item {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.7rem;
    background: #faf7f2;
    border: 1px solid #ede5d8;
    border-radius: 6px;
    margin-top: 0.4rem;
    font-size: 0.74rem;
    color: #3d2f22;
}
.file-ok { color: #2d7a3a; font-weight: 600; font-size: 0.65rem; }
.file-err { color: #c0392b; font-weight: 600; font-size: 0.65rem; }

/* Override Streamlit buttons */
.stButton > button {
    background: #c8701a !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'Instrument Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.8rem !important;
    padding: 0.5rem 1rem !important;
    width: 100% !important;
    box-shadow: 0 2px 8px rgba(200,112,26,0.25) !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    background: #a85c10 !important;
    box-shadow: 0 4px 12px rgba(200,112,26,0.35) !important;
}

/* Danger button */
.danger-btn > button {
    background: transparent !important;
    color: #9c8878 !important;
    border: 1px solid #ede5d8 !important;
    box-shadow: none !important;
    font-size: 0.74rem !important;
}
.danger-btn > button:hover {
    color: #c0392b !important;
    border-color: #c0392b !important;
    background: #fdecea !important;
}

/* Input overrides */
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #ffffff !important;
    border: 1px solid #ede5d8 !important;
    border-radius: 8px !important;
    color: #1c1612 !important;
    font-family: 'Instrument Sans', sans-serif !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #c8701a !important;
    box-shadow: 0 0 0 3px rgba(200,112,26,0.12) !important;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: #ffffff !important;
    border-right: 1px solid #ede5d8 !important;
}
section[data-testid="stSidebar"] .block-container {
    padding: 1rem 0.75rem !important;
}

/* File uploader */
[data-testid="stFileUploader"] {
    background: #faf7f2 !important;
    border: 1.5px dashed #d4c5b0 !important;
    border-radius: 8px !important;
}
</style>
""", unsafe_allow_html=True)


# ── Helper functions ──────────────────────────────────────────────────────────

def get_stats():
    try:
        r = requests.get(f"{API}/stats", timeout=3)
        return r.json() if r.ok else None
    except:
        return None

def ingest_pdf_file(file_bytes, filename):
    try:
        r = requests.post(
            f"{API}/ingest/pdf",
            files={"file": (filename, file_bytes, "application/pdf")},
            timeout=60
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def ingest_text_content(text, source):
    try:
        r = requests.post(
            f"{API}/ingest/text",
            json={"text": text, "source": source},
            timeout=30
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def query_rag(question, history):
    try:
        r = requests.post(
            f"{API}/query",
            json={"query": question, "chat_history": history},
            timeout=60
        )
        return r.json()
    except Exception as e:
        return {"error": str(e)}

def clear_index():
    try:
        r = requests.delete(f"{API}/clear", timeout=10)
        return r.json()
    except Exception as e:
        return {"error": str(e)}


# ── Session state ─────────────────────────────────────────────────────────────
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "api_history" not in st.session_state:
    st.session_state.api_history = []
if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []


# ── Fetch stats ───────────────────────────────────────────────────────────────
stats = get_stats()
api_online = stats is not None


# ── Header ────────────────────────────────────────────────────────────────────
api_status = "🟢 API Online" if api_online else "🔴 API Offline"
chunks = stats["total_chunks"] if stats else 0
docs = stats["total_documents"] if stats else 0

st.markdown(f"""
<div class="docmind-header">
    <div>
        <div class="docmind-logo">📖 Doc<span>Mind</span></div>
        <div class="docmind-tagline">RAG Document Intelligence · Powered by Groq Llama 3 (Free)</div>
    </div>
    <div class="stat-row">
        <span class="{'status-online' if api_online else 'status-offline'}">{api_status}</span>
        <span class="stat-badge">⬡ {chunks} chunks</span>
        <span class="stat-badge">📄 {docs} docs</span>
    </div>
</div>
""", unsafe_allow_html=True)

if not api_online:
    st.error("⚠️ Backend not running. Open a terminal and run: `uvicorn backend.api:app --reload --port 8000`")


# ── Layout ────────────────────────────────────────────────────────────────────
sidebar, main_col = st.columns([1, 2.5])


# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with sidebar:

    # PDF Upload
    st.markdown('<div class="section-label">📄 Upload PDF</div>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Drop PDF here",
        type=["pdf"],
        label_visibility="collapsed"
    )

    if uploaded_file:
        if st.button("📤 Index This PDF"):
            if not api_online:
                st.error("API is offline.")
            else:
                with st.spinner(f"Indexing {uploaded_file.name}..."):
                    result = ingest_pdf_file(uploaded_file.read(), uploaded_file.name)
                if "error" in result:
                    st.error(f"Failed: {result['error']}")
                    st.session_state.indexed_files.append({"name": uploaded_file.name, "status": "error", "chunks": 0})
                else:
                    st.success(f"✅ {result['chunks_added']} chunks indexed!")
                    st.session_state.indexed_files.append({"name": uploaded_file.name, "status": "ok", "chunks": result['chunks_added']})
                    st.rerun()

    # Show indexed files
    if st.session_state.indexed_files:
        for f in st.session_state.indexed_files[-5:]:
            icon = "✓" if f["status"] == "ok" else "✗"
            cls = "file-ok" if f["status"] == "ok" else "file-err"
            st.markdown(f"""
            <div class="file-item">
                📄 <span style="flex:1;overflow:hidden;text-overflow:ellipsis;">{f['name']}</span>
                <span class="{cls}">{icon} {f['chunks']} chunks</span>
            </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Text ingest
    st.markdown('<div class="section-label">✏️ Paste Text</div>', unsafe_allow_html=True)
    source_name = st.text_input("Source name", value="notes", placeholder="e.g. lecture_notes")
    text_input = st.text_area("Content", height=100, placeholder="Paste any text here…", label_visibility="collapsed")

    if st.button("📤 Index Text"):
        if not api_online:
            st.error("API is offline.")
        elif len(text_input.strip()) < 50:
            st.warning("Text too short (min 50 characters)")
        else:
            with st.spinner("Indexing..."):
                result = ingest_text_content(text_input, source_name)
            if "error" in result:
                st.error(result["error"])
            else:
                st.success(f"✅ {result['chunks_added']} chunks indexed!")
                st.session_state.indexed_files.append({"name": source_name, "status": "ok", "chunks": result['chunks_added']})
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Indexed sources
    if stats and stats["sources"]:
        st.markdown('<div class="section-label">🗂 Indexed Sources</div>', unsafe_allow_html=True)
        for src in stats["sources"]:
            st.markdown(f'<div class="file-item">📄 {src}</div>', unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

    # Clear
    st.markdown('<div class="danger-btn">', unsafe_allow_html=True)
    if st.button("🗑 Clear All Documents"):
        clear_index()
        st.session_state.chat_history = []
        st.session_state.api_history = []
        st.session_state.indexed_files = []
        st.success("Cleared!")
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)


# ── MAIN CHAT ─────────────────────────────────────────────────────────────────
with main_col:

    st.markdown('<div class="section-label">💬 Conversation</div>', unsafe_allow_html=True)

    # Chat messages
    if not st.session_state.chat_history:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-icon">💬</div>
            <div class="empty-title">Start a conversation</div>
            <div class="empty-sub">Upload a PDF or paste text in the sidebar, then ask anything about it.</div>
            <div class="chips-row">
                <span class="chip-item">Summarize the main points</span>
                <span class="chip-item">What are the key findings?</span>
                <span class="chip-item">List all topics covered</span>
                <span class="chip-item">Give me a brief overview</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.chat_history:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="chat-label-user">YOU</div>
                <div class="chat-user">{msg['content']}</div>
                """, unsafe_allow_html=True)
            else:
                sources_html = "".join(
                    f'<span class="source-pill">📄 {s}</span>'
                    for s in msg.get("sources", [])
                )
                meta = f'<div class="chat-meta">⬡ {msg.get("chunks_used", 0)} chunks retrieved &nbsp;·&nbsp; {sources_html}</div>' if msg.get("sources") else ""
                content = msg["content"].replace("\n", "<br>")
                st.markdown(f"""
                <div class="chat-label-bot">DOCMIND</div>
                <div class="chat-bot">{content}{meta}</div>
                """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Query input
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_query = st.text_input(
            "Ask a question",
            placeholder="Ask anything about your documents…",
            label_visibility="collapsed",
            key="query_input"
        )
    with col_btn:
        ask = st.button("Ask →")

    # Example chips as buttons
    if not st.session_state.chat_history:
        c1, c2, c3 = st.columns(3)
        with c1:
            if st.button("Summarize main points"):
                user_query = "Summarize the main points"
                ask = True
        with c2:
            if st.button("What are key findings?"):
                user_query = "What are the key findings?"
                ask = True
        with c3:
            if st.button("List all topics covered"):
                user_query = "List all topics covered"
                ask = True

    # Process query
    if ask and user_query:
        if not api_online:
            st.error("Start the backend first!")
        else:
            with st.spinner("Thinking…"):
                result = query_rag(user_query, st.session_state.api_history)

            if "error" in result:
                st.error(f"Error: {result['error']}")
            else:
                st.session_state.chat_history.append({"role": "user", "content": user_query})
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "chunks_used": result.get("chunks_used", 0)
                })
                st.session_state.api_history.append({"role": "user", "content": user_query})
                st.session_state.api_history.append({"role": "assistant", "content": result["answer"]})
                if len(st.session_state.api_history) > 12:
                    st.session_state.api_history = st.session_state.api_history[-12:]
                st.rerun()
