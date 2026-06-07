"""
DocMind — Streamlit Frontend
Works in two modes:
  - Local (default)   : talks to FastAPI backend at DOCMIND_API_URL
  - Standalone Cloud  : set DOCMIND_STANDALONE=true — RAG runs in-process
"""

import os
import sys
from pathlib import Path

import streamlit as st

STANDALONE = os.getenv("DOCMIND_STANDALONE", "false").lower() == "true"
API_URL    = os.getenv("DOCMIND_API_URL", "http://localhost:8000").rstrip("/")

st.set_page_config(
    page_title="DocMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap');

html, body,
[data-testid="stApp"],
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.main, .block-container, [class*="css"] {
    background-color: #F0F4FF !important;
    color: #0F172A !important;
    font-family: 'Inter', sans-serif !important;
}
.block-container {
    padding: 2rem 2.5rem 4rem !important;
    max-width: 1080px !important;
}

#MainMenu, footer, header { visibility: hidden !important; }
[data-testid="stDecoration"] { display: none !important; }

[data-testid="stSidebar"],
[data-testid="stSidebar"] > div,
[data-testid="stSidebar"] section {
    background-color: #FFFFFF !important;
    border-right: 1px solid #DBEAFE !important;
}
[data-testid="stSidebar"] > div:first-child { padding: 1.5rem 1.25rem 2rem !important; }
[data-testid="stSidebar"] *, [data-testid="stSidebar"] p,
[data-testid="stSidebar"] span, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] div { color: #0F172A !important; }

[data-testid="stRadio"] label {
    color: #0F172A !important; font-size: 14px !important; font-weight: 500 !important;
}
[data-testid="stRadio"] > div { gap: 6px !important; }

textarea, input[type="text"], input[type="search"] {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1.5px solid #BFDBFE !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 14px !important;
}
textarea:focus, input:focus {
    border-color: #2563EB !important;
    box-shadow: 0 0 0 3px rgba(37,99,235,0.1) !important;
    outline: none !important;
}

[data-testid="stFileUploader"] {
    background-color: #EFF6FF !important;
    border: 2px dashed #93C5FD !important;
    border-radius: 10px !important;
}
[data-testid="stFileUploader"] * { color: #1D4ED8 !important; }

[data-testid="stBaseButton-secondary"], button[kind="secondary"] {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    font-family: 'Inter', sans-serif !important;
}
[data-testid="stBaseButton-secondary"]:hover { background-color: #1D4ED8 !important; }

[data-testid="stTabs"] { background: transparent !important; }
[data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 2px solid #DBEAFE !important;
    gap: 0 !important;
}
[data-baseweb="tab"] {
    background: transparent !important;
    color: #64748B !important;
    font-size: 14px !important;
    font-weight: 500 !important;
    padding: 0.6rem 1.25rem !important;
    border-bottom: 2px solid transparent !important;
    margin-bottom: -2px !important;
}
[data-baseweb="tab"][aria-selected="true"] {
    color: #2563EB !important;
    border-bottom: 2px solid #2563EB !important;
}
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display: none !important; }
[data-testid="stTabsContent"] { background: transparent !important; padding-top: 1.5rem !important; }

[data-testid="stChatMessage"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DBEAFE !important;
    border-radius: 10px !important;
    padding: 0.9rem 1.1rem !important;
    margin-bottom: 0.75rem !important;
}
[data-testid="stChatMessage"] * { color: #0F172A !important; }

/* Fix chat input — force light */
[data-testid="stChatInput"],
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea {
    background-color: #FFFFFF !important;
    color: #0F172A !important;
    border: 1.5px solid #BFDBFE !important;
    border-radius: 10px !important;
}
[data-testid="stChatInput"] button {
    background-color: #2563EB !important;
    border-radius: 8px !important;
}
[data-testid="stBottomBlockContainer"],
[data-testid="stBottomBlockContainer"] > div {
    background-color: #F0F4FF !important;
}

[data-testid="stAlert"] { border-radius: 8px !important; font-size: 14px !important; }
[data-testid="stExpander"] {
    background-color: #FFFFFF !important;
    border: 1px solid #DBEAFE !important;
    border-radius: 8px !important;
}

::-webkit-scrollbar { width: 5px; }
::-webkit-scrollbar-track { background: #F0F4FF; }
::-webkit-scrollbar-thumb { background: #BFDBFE; border-radius: 10px; }
hr { border: none !important; border-top: 1px solid #DBEAFE !important; }
</style>
""", unsafe_allow_html=True)


# ── Standalone bootstrap ───────────────────────────────────────────────────────
if STANDALONE:
    os.environ["PERSIST_INDEX"] = "false"
    ROOT = Path(__file__).resolve().parent
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    import backend.rag_engine as engine
    if "rag_index" not in st.session_state:
        import faiss as _faiss
        st.session_state.rag_index    = _faiss.IndexFlatIP(engine.EMBED_DIM)
        st.session_state.rag_metadata = []
        engine._index    = st.session_state.rag_index
        engine._metadata = st.session_state.rag_metadata
    else:
        engine._index    = st.session_state.rag_index
        engine._metadata = st.session_state.rag_metadata


def _api(method, path, **kwargs):
    import requests
    try:
        r = requests.request(method, f"{API_URL}{path}", timeout=60, **kwargs)
        r.raise_for_status()
        return r.json()
    except requests.exceptions.ConnectionError:
        st.error(f"Cannot reach backend at {API_URL}. Run: uvicorn backend.api:app --reload --port 8000")
        st.stop()

def get_stats():
    return engine.get_stats() if STANDALONE else _api("GET", "/stats")


# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:

    # Logo
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;margin-bottom:1.75rem;">
        <div style="width:36px;height:36px;background:linear-gradient(135deg,#2563EB,#60A5FA);
                    border-radius:9px;display:flex;align-items:center;justify-content:center;font-size:18px;">🧠</div>
        <div>
            <div style="font-size:15px;font-weight:600;color:#0F172A;">DocMind</div>
            <div style="font-size:11px;color:#64748B;">RAG Document Intelligence</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Step indicator
    stats        = get_stats()
    total_chunks = stats.get("total_chunks", 0)
    total_docs   = stats.get("total_documents", 0)
    sources      = stats.get("sources", [])

    step1_done = total_chunks > 0
    st.markdown(f"""
    <div style="margin-bottom:1.25rem;">
        <div style="display:flex;align-items:center;gap:8px;padding:8px 10px;
                    background:{'#EFF6FF' if step1_done else '#F8FAFC'};
                    border:1px solid {'#BFDBFE' if step1_done else '#E2E8F0'};
                    border-radius:7px;margin-bottom:6px;">
            <div style="width:20px;height:20px;border-radius:50%;background:{'#2563EB' if step1_done else '#CBD5E1'};
                        display:flex;align-items:center;justify-content:center;
                        font-size:11px;font-weight:700;color:#fff;flex-shrink:0;">
                {'✓' if step1_done else '1'}
            </div>
            <div style="font-size:12.5px;font-weight:500;color:{'#1D4ED8' if step1_done else '#64748B'};">
                {'Document indexed' if step1_done else 'Upload a document'}
            </div>
        </div>
        <div style="display:flex;align-items:center;gap:8px;padding:8px 10px;
                    background:{'#EFF6FF' if step1_done else '#F8FAFC'};
                    border:1px solid {'#BFDBFE' if step1_done else '#E2E8F0'};
                    border-radius:7px;">
            <div style="width:20px;height:20px;border-radius:50%;background:{'#2563EB' if step1_done else '#CBD5E1'};
                        display:flex;align-items:center;justify-content:center;
                        font-size:11px;font-weight:700;color:#fff;flex-shrink:0;">2</div>
            <div style="font-size:12.5px;font-weight:500;color:{'#1D4ED8' if step1_done else '#64748B'};">
                Ask questions in Chat
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<hr>", unsafe_allow_html=True)

    # Chunking strategy
    st.markdown('<div style="font-size:10.5px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#2563EB;margin:0.75rem 0 0.5rem;">Chunking Strategy</div>', unsafe_allow_html=True)

    strategy = st.radio(
        "strategy_radio",
        options=["fixed", "sentence"],
        format_func=lambda x: "Fixed (500 chars)" if x == "fixed" else "Sentence-aware  ✦ recommended",
        label_visibility="collapsed",
    )

    if strategy == "fixed":
        st.markdown('<div style="background:#F8FAFC;border-left:3px solid #94A3B8;border-radius:0 6px 6px 0;padding:7px 10px;font-size:12px;color:#475569;line-height:1.5;margin-top:4px;">Splits every 500 chars. Fast, may cut sentences mid-way.</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div style="background:#EFF6FF;border-left:3px solid #2563EB;border-radius:0 6px 6px 0;padding:7px 10px;font-size:12px;color:#1E40AF;line-height:1.5;margin-top:4px;">Complete sentences, semantic overlap. <strong>+10% Recall@3</strong> on benchmark.</div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin:1rem 0;'>", unsafe_allow_html=True)

    # Index stats
    st.markdown('<div style="font-size:10.5px;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;color:#2563EB;margin-bottom:0.6rem;">Index</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin-bottom:0.6rem;">
        <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;padding:9px 12px;">
            <div style="font-size:20px;font-weight:600;color:#1D4ED8;font-family:monospace;">{total_chunks}</div>
            <div style="font-size:10px;color:#3B82F6;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Chunks</div>
        </div>
        <div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;padding:9px 12px;">
            <div style="font-size:20px;font-weight:600;color:#1D4ED8;font-family:monospace;">{total_docs}</div>
            <div style="font-size:10px;color:#3B82F6;font-weight:600;text-transform:uppercase;letter-spacing:0.05em;">Docs</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if sources:
        tags = "".join(f'<span style="display:inline-block;font-size:10.5px;background:#DBEAFE;color:#1E40AF;border-radius:4px;padding:2px 7px;margin:2px 2px 0 0;font-family:monospace;max-width:170px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;">{s}</span>' for s in sources)
        st.markdown(f'<div style="margin-bottom:0.5rem;">{tags}</div>', unsafe_allow_html=True)

    st.markdown("<hr style='margin:0.75rem 0;'>", unsafe_allow_html=True)

    if st.button("🗑  Clear index", use_container_width=True):
        if STANDALONE:
            engine.clear_index()
            st.session_state.rag_index    = engine._index
            st.session_state.rag_metadata = engine._metadata
        else:
            _api("DELETE", "/clear")
        st.session_state.chat_history = []
        st.rerun()

    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
    badge_bg  = "#D1FAE5" if STANDALONE else "#DBEAFE"
    badge_col = "#065F46" if STANDALONE else "#1E40AF"
    badge_txt = "Standalone mode" if STANDALONE else f"Connected · {API_URL}"
    st.markdown(f'<div style="font-size:11px;font-weight:500;background:{badge_bg};color:{badge_col};padding:4px 10px;border-radius:5px;display:inline-block;">● {badge_txt}</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="border-bottom:2px solid #DBEAFE;padding-bottom:1rem;margin-bottom:1.5rem;">
    <h1 style="font-size:26px;font-weight:600;color:#0F172A;letter-spacing:-0.03em;margin:0 0 5px;">DocMind</h1>
    <p style="font-size:13.5px;color:#64748B;margin:0;line-height:1.55;">
        Upload a document → choose a chunking strategy → ask questions.
        Answers are grounded in your document and cited using
        <strong style="color:#1D4ED8;">Llama 3.3 70B via Groq</strong>.
    </p>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab_ingest, tab_chat = st.tabs(["📄  Step 1 — Ingest document", "💬  Step 2 — Chat"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — INGEST
# ══════════════════════════════════════════════════════════════════════════════
with tab_ingest:

    col1, col2 = st.columns(2, gap="large")

    # LEFT — PDF
    with col1:
        st.markdown('<div style="font-size:13px;font-weight:600;color:#0F172A;margin-bottom:0.6rem;">Upload a PDF <span style="font-size:11px;font-weight:400;color:#94A3B8;">· max 20 MB</span></div>', unsafe_allow_html=True)
        uploaded = st.file_uploader("pdf", type=["pdf"], label_visibility="collapsed")

        if uploaded:
            kb = len(uploaded.getvalue()) / 1024
            st.markdown(f'<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:7px;padding:7px 12px;font-size:12px;color:#1E40AF;font-family:monospace;margin:6px 0;">{uploaded.name} · {kb:.1f} KB</div>', unsafe_allow_html=True)
            if st.button("Ingest PDF →", use_container_width=True, key="btn_pdf"):
                with st.spinner(f"Processing with {strategy} chunking…"):
                    if STANDALONE:
                        result = engine.ingest_pdf_bytes(uploaded.read(), uploaded.name, strategy=strategy)
                    else:
                        result = _api("POST", f"/ingest/pdf?strategy={strategy}",
                                      files={"file": (uploaded.name, uploaded.getvalue(), "application/pdf")})
                st.success(f"✓ {result.get('chunks_added','?')} chunks added · {result.get('strategy', strategy)} chunking · now go to Chat →")
                st.rerun()
        else:
            st.markdown('<div style="text-align:center;padding:1.25rem;color:#93C5FD;font-size:13px;">Drop a PDF here or click Browse files</div>', unsafe_allow_html=True)

    # RIGHT — Text
    with col2:
        st.markdown('<div style="font-size:13px;font-weight:600;color:#0F172A;margin-bottom:0.6rem;">Paste text <span style="font-size:11px;font-weight:400;color:#94A3B8;">· min 50 chars</span></div>', unsafe_allow_html=True)
        raw_text = st.text_area("text", placeholder="Paste an article, documentation, notes, or any text here…", height=130, label_visibility="collapsed")
        source   = st.text_input("source", value="pasted_text", placeholder="Source label", label_visibility="collapsed")
        if st.button("Ingest text →", use_container_width=True, key="btn_text"):
            if len(raw_text.strip()) < 50:
                st.warning("Paste at least 50 characters.")
            else:
                with st.spinner(f"Processing with {strategy} chunking…"):
                    if STANDALONE:
                        result = engine.ingest_text(raw_text, source=source, strategy=strategy)
                    else:
                        result = _api("POST", f"/ingest/text?strategy={strategy}",
                                      json={"text": raw_text, "source": source})
                st.success(f"✓ {result.get('chunks_added','?')} chunks added · {result.get('strategy', strategy)} chunking · now go to Chat →")
                st.rerun()

    # Benchmark — collapsed by default, visible but not intrusive
    st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
    st.markdown('<div style="font-size:10.5px;font-weight:600;letter-spacing:0.09em;text-transform:uppercase;color:#3B82F6;margin-bottom:0.6rem;">Retrieval Benchmark</div>', unsafe_allow_html=True)
    bc1, bc2, bc3 = st.columns(3)
    with bc1:
        st.markdown('<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;padding:10px 14px;text-align:center;"><div style="font-size:9.5px;font-weight:600;text-transform:uppercase;color:#94A3B8;letter-spacing:0.06em;margin-bottom:4px;">Fixed · Recall@3</div><div style="font-size:24px;font-weight:600;color:#64748B;font-family:monospace;">70%</div></div>', unsafe_allow_html=True)
    with bc2:
        st.markdown('<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;padding:10px 14px;text-align:center;"><div style="font-size:9.5px;font-weight:600;text-transform:uppercase;color:#60A5FA;letter-spacing:0.06em;margin-bottom:4px;">Sentence · Recall@3</div><div style="font-size:24px;font-weight:600;color:#1D4ED8;font-family:monospace;">80%</div></div>', unsafe_allow_html=True)
    with bc3:
        st.markdown('<div style="background:#EFF6FF;border:1px solid #BFDBFE;border-radius:8px;padding:10px 14px;text-align:center;"><div style="font-size:9.5px;font-weight:600;text-transform:uppercase;color:#60A5FA;letter-spacing:0.06em;margin-bottom:4px;">Sentence · Recall@5</div><div style="font-size:24px;font-weight:600;color:#1D4ED8;font-family:monospace;">90%</div></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-size:11px;color:#94A3B8;margin-top:0.6rem;">Run <code style="background:#F1F5F9;color:#475569;padding:1px 5px;border-radius:3px;font-size:10.5px;">python eval/run_eval.py --verbose</code> to reproduce · no API keys needed</div>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — CHAT
# ══════════════════════════════════════════════════════════════════════════════
with tab_chat:

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Empty state — no docs
    if total_chunks == 0:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;border:2px dashed #BFDBFE;
                    border-radius:12px;background:#F8FBFF;">
            <div style="font-size:30px;margin-bottom:0.6rem;">📭</div>
            <div style="font-size:15px;font-weight:600;color:#1E40AF;margin-bottom:6px;">No documents yet</div>
            <div style="font-size:13px;color:#64748B;line-height:1.6;">
                Go to <strong>Step 1 — Ingest document</strong>, upload a PDF or paste text,
                then come back here to ask questions.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # Ready state — no chat yet
        if not st.session_state.chat_history:
            st.markdown(f"""
            <div style="text-align:center;padding:2rem 1rem;border:1px solid #BFDBFE;
                        border-radius:10px;background:#EFF6FF;margin-bottom:1rem;">
                <div style="font-size:24px;margin-bottom:0.5rem;">🔍</div>
                <div style="font-size:14px;font-weight:600;color:#1D4ED8;margin-bottom:3px;">
                    Ready · {total_chunks} chunks indexed across {total_docs} document{'s' if total_docs != 1 else ''}
                </div>
                <div style="font-size:12.5px;color:#3B82F6;">Type your question below to get started.</div>
            </div>
            """, unsafe_allow_html=True)

        # Chat history
        for msg in st.session_state.chat_history:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if msg["role"] == "assistant" and msg.get("sources"):
                    chips = "".join(
                        f'<span style="display:inline-flex;align-items:center;gap:4px;font-size:11px;'
                        f'background:#EFF6FF;color:#1E40AF;border:1px solid #BFDBFE;border-radius:4px;'
                        f'padding:2px 7px;margin:3px 2px 0 0;font-family:monospace;">'
                        f'<span style="width:5px;height:5px;background:#3B82F6;border-radius:50%;display:inline-block;"></span>{s}</span>'
                        for s in msg["sources"]
                    )
                    st.markdown(
                        f'<div style="margin-top:8px;">{chips}'
                        f'<span style="font-size:11px;color:#94A3B8;margin-left:6px;">'
                        f'{msg.get("chunks_used","")} chunks retrieved</span></div>',
                        unsafe_allow_html=True,
                    )

        # Input
        if question := st.chat_input("Ask a question about your document…"):
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Retrieving and generating…"):
                    if STANDALONE:
                        chunks      = engine.retrieve(question)
                        answer      = engine.generate_answer(question, chunks,
                                        chat_history=st.session_state.chat_history[:-1])
                        sources     = list({c["source"] for c in chunks})
                        chunks_used = len(chunks)
                    else:
                        data        = _api("POST", "/query", json={
                                        "query": question,
                                        "chat_history": st.session_state.chat_history[:-1]})
                        answer      = data.get("answer", "No answer returned.")
                        sources     = data.get("sources", [])
                        chunks_used = data.get("chunks_used", 0)

                st.markdown(answer)
                if sources:
                    chips = "".join(
                        f'<span style="display:inline-flex;align-items:center;gap:4px;font-size:11px;'
                        f'background:#EFF6FF;color:#1E40AF;border:1px solid #BFDBFE;border-radius:4px;'
                        f'padding:2px 7px;margin:3px 2px 0 0;font-family:monospace;">'
                        f'<span style="width:5px;height:5px;background:#3B82F6;border-radius:50%;display:inline-block;"></span>{s}</span>'
                        for s in sources
                    )
                    st.markdown(
                        f'<div style="margin-top:8px;">{chips}'
                        f'<span style="font-size:11px;color:#94A3B8;margin-left:6px;">{chunks_used} chunks retrieved</span></div>',
                        unsafe_allow_html=True,
                    )

            st.session_state.chat_history.append({
                "role": "assistant", "content": answer,
                "sources": sources, "chunks_used": chunks_used,
            })