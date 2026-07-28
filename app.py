"""
KhmerSME Knowledge Search — RAG-Based AI Search System (CS382 Final Project)

Run locally with:
    streamlit run app.py
"""

import os
import time
from dotenv import load_dotenv
import streamlit as st

# 1. Load local .env file if running on a local machine
load_dotenv()

# 2. Safely load secrets if running on Streamlit Community Cloud
try:
    if hasattr(st, "secrets"):
        for key, value in st.secrets.items():
            os.environ[key] = str(value)
except Exception:
    # Ignores missing secrets file when running locally
    pass

from rag.embed_store import VectorStore
from rag.generate import RELEVANCE_THRESHOLD, generate_answer
from rag.ingest import build_chunk_records, load_documents

DATA_FOLDER = "data/sme_docs"

# ---------------------------- Page Config & Custom CSS ----------------------------
st.set_page_config(
    page_title="KhmerSME Knowledge Search", 
    page_icon="🔎", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for UI refinement
st.markdown("""
<style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .hero-container {
        padding: 1.5rem 0rem 1rem 0rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-bottom: 1.5rem;
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.2rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .hero-subtitle {
        color: #A0AEC0;
        font-size: 0.95rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        background-color: rgba(72, 187, 120, 0.2);
        color: #48BB78;
        border: 1px solid rgba(72, 187, 120, 0.4);
    }
    .source-card {
        background-color: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #4299E1;
        padding: 0.8rem 1rem;
        border-radius: 0px 8px 8px 0px;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner="Loading, chunking, and embedding documents...")
def load_store(backend: str, chunk_size: int):
    docs = load_documents(DATA_FOLDER)
    chunks = build_chunk_records(docs, chunk_size=chunk_size)
    store = VectorStore(backend=backend)
    store.build(chunks)
    return store, docs, chunks


# ---------------------------- Sidebar Layout ----------------------------
with st.sidebar:
    st.markdown("## ⚙️ **Control Panel**")
    st.caption("Configure parameters for vector search and generation.")
    
    with st.expander("🛠️ **Model & Retrieval Settings**", expanded=True):
        mode = st.radio(
            "Answer Mode", 
            ["llm", "extractive"], 
            index=0,
            help="LLM mode uses Groq for synthesis. Extractive mode returns raw passages."
        )
        top_k = st.slider("Retrieval Count (Top-K)", min_value=1, max_value=10, value=3)
        chunk_size = st.select_slider("Chunk Size (Words)", options=[80, 120, 160, 200], value=120)
        backend = st.selectbox(
            "Embedding Model", 
            ["auto", "st", "tfidf"], 
            index=0,
            help="Select the vector embedding engine."
        )

store, docs, chunks = load_store(backend, chunk_size)
threshold = getattr(store.backend, "default_threshold", RELEVANCE_THRESHOLD)

with st.sidebar:
    st.markdown("---")
    st.markdown("### 📊 **Index Metadata**")
    col_a, col_b = st.columns(2)
    col_a.metric("Documents", len(docs))
    col_b.metric("Chunks", len(chunks))
    
    st.caption(f"**Embedding:** `{store.backend.name}`")
    st.caption(f"**Relevance Threshold:** `{threshold}`")
    
    with st.expander("📚 **Indexed Files**", expanded=False):
        for d in docs:
            st.markdown(f"- 📄 `{d['title']}`")

# ------------------------------- Hero Section --------------------------------
st.markdown("""
<div class="hero-container">
    <div class="hero-title">
        <span>🔎 KhmerSME Knowledge Search</span>
        <span class="status-badge">● Active RAG</span>
    </div>
    <div class="hero-subtitle">
        Intelligent search assistant grounded in Cambodian SME policy, digital economy strategies, and business regulations.
    </div>
</div>
""", unsafe_allow_html=True)

# Initialize Chat Session State
if "messages" not in st.session_state:
    st.session_state.messages = []

# Helper function to render execution metrics
def render_metrics(t_retrieve, t_generate, mode_name, backend_name):
    with st.container():
        c1, c2, c3, c4 = st.columns(4)
        c1.caption(f"⚡ **Retrieval:** `{t_retrieve * 1000:.0f} ms`")
        c2.caption(f"🧠 **Generation:** `{t_generate:.2f} s`")
        c3.caption(f"⚙️ **Mode:** `{mode_name}`")
        c4.caption(f"📦 **Backend:** `{backend_name}`")

# Render existing chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("meta"):
            meta = message["meta"]
            with st.container(border=True):
                render_metrics(meta['t_retrieve'], meta['t_generate'], meta['mode'], meta['backend'])
            
            if meta.get("sources"):
                with st.expander("📚 **Cited Sources & Context**"):
                    for i, (chunk, score) in enumerate(meta["sources"], start=1):
                        marker = "✅" if score >= threshold else "⚠️"
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <b>{marker} [{i}] {chunk.doc_title}</b> &nbsp;|&nbsp; <code>Score: {score:.2f}</code><br>
                                <p style="margin-top: 0.4rem; font-size: 0.9rem;">{chunk.text}</p>
                                <span style="font-size: 0.75rem; color: #A0AEC0;">File: {chunk.source_file} | Chunk: {chunk.chunk_id}</span>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )

# Chat Input & Assistant Response
if prompt := st.chat_input("Ask a question about Cambodian SMEs, business laws, or general topics..."):
    
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing documents & generating response..."):
            t0 = time.perf_counter()
            retrieved = store.query(prompt, top_k=top_k)
            t_retrieve = time.perf_counter() - t0

            t1 = time.perf_counter()
            answer = generate_answer(prompt, retrieved, mode=mode, threshold=threshold)
            t_generate = time.perf_counter() - t1

            has_relevant_sources = bool(retrieved) and retrieved[0][1] >= threshold

            # Output answer
            st.markdown(answer)

            # Output Execution Metadata Card
            with st.container(border=True):
                render_metrics(t_retrieve, t_generate, mode, store.backend.name)

            meta_data = {
                "t_retrieve": t_retrieve,
                "t_generate": t_generate,
                "mode": mode,
                "backend": store.backend.name,
                "sources": retrieved if has_relevant_sources else None
            }

            # Display Citation Cards
            if has_relevant_sources:
                with st.expander("📚 **Cited Sources & Context**"):
                    for i, (chunk, score) in enumerate(retrieved, start=1):
                        marker = "✅" if score >= threshold else "⚠️"
                        st.markdown(
                            f"""
                            <div class="source-card">
                                <b>{marker} [{i}] {chunk.doc_title}</b> &nbsp;|&nbsp; <code>Score: {score:.2f}</code><br>
                                <p style="margin-top: 0.4rem; font-size: 0.9rem;">{chunk.text}</p>
                                <span style="font-size: 0.75rem; color: #A0AEC0;">File: {chunk.source_file} | Chunk: {chunk.chunk_id}</span>
                            </div>
                            """, 
                            unsafe_allow_html=True
                        )
            else:
                st.caption("ℹ️ *No direct document citations were required or matched for this query.*")

    st.session_state.messages.append({
        "role": "assistant", 
        "content": answer,
        "meta": meta_data
    })