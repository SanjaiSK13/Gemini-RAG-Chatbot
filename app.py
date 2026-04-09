from dotenv import load_dotenv
import streamlit as st
import os
import time
from src.data_loader import load_and_split_data
from src.embedding_engine import create_vector_store, load_vector_store
from src.rag_chain import get_rag_chain
from src.config import LLM_MODEL
from src.evaluator import compute_contextual_relevance, save_feedback, get_metrics_summary

load_dotenv()

# 1. Page Identity
st.set_page_config(page_title="Sanjai Support AI", page_icon="🛡️", layout="wide")

# 2. Professional CSS (Kept exactly as provided)
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; }
    .main-header { color: #1E3A8A; font-size: 2.2rem; font-weight: 700; margin-bottom: 0px; }
    .status-text { font-size: 0.8rem; color: #6C757D; }
    .stChatMessage { border-radius: 12px; border: 1px solid #E9ECEF; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    .metric-pill { display:inline-block; padding:3px 10px; border-radius:99px; font-size:12px; font-weight:600; }
    </style>
    """, unsafe_allow_html=True)

# 3. High-Speed Caching (Optimized for Sub-2s Latency)
@st.cache_resource(show_spinner=False)
def init_system():
    if not os.path.exists("vector_store"):
        os.makedirs("vector_store")
    # Check for the actual index file to prevent re-processing
    if not os.path.exists("vector_store/index.faiss"): 
        chunks = load_and_split_data()
        vs = create_vector_store(chunks)
    else:
        vs = load_vector_store()
    return get_rag_chain(vs)

# 4. Sidebar (Kept exactly as provided)
with st.sidebar:
    st.markdown("### ⚙️ System Profile")
    st.success(f"**Model:** {LLM_MODEL}")
    st.info("**Database:** FAISS Vector Index")
    st.markdown("---")

    st.markdown("### 📊 Session Metrics")
    metrics = get_metrics_summary(st.session_state.get("eval_log", []))

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Avg Latency", f"{metrics['avg_latency']:.2f}s")
        st.metric("Queries", metrics["total_queries"])
    with col2:
        st.metric("Relevance", f"{metrics['avg_relevance']:.0%}")
        st.metric("👍 Rate", f"{metrics['satisfaction_rate']:.0%}")

    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.eval_log = []
        st.rerun()

# 5. UI Header (Kept exactly as provided)
st.markdown('<p class="main-header">🛡️ Intelligence Support Portal</p>', unsafe_allow_html=True)
st.markdown('<p class="status-text">Grounded RAG Pipeline | Security Verified</p>', unsafe_allow_html=True)

if not st.session_state.get("messages"):
    st.markdown("""
    <div style="background:#FFFFFF; border:1px solid #E9ECEF; border-radius:12px; padding:20px 24px; margin:16px 0;">
        <p style="font-size:13px; color:#6C757D; margin:0 0 4px; text-transform:uppercase; letter-spacing:0.05em;">What this chatbot does</p>
        <p style="font-size:16px; font-weight:600; color:#1E3A8A; margin:0 0 10px;">
            Your AI-powered customer support assistant — grounded in your own documents.
        </p>
        <p style="font-size:14px; color:#495057; margin:0 0 16px;">
            Ask anything about our products, policies, or services. Every answer is
            <strong>retrieved from a live knowledge base</strong> — not guessed.
            Powered by <strong>Gemini LLM</strong> + <strong>FAISS vector search</strong>.
        </p>
        <div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px;">
            <span style="background:#E6F1FB; color:#185FA5; font-size:12px; padding:4px 10px; border-radius:99px; font-weight:500;">📄 Document-aware</span>
            <span style="background:#EAF3DE; color:#3B6D11; font-size:12px; padding:4px 10px; border-radius:99px; font-weight:500;">⚡ Low latency</span>
            <span style="background:#FAEEDA; color:#854F0B; font-size:12px; padding:4px 10px; border-radius:99px; font-weight:500;">🛡️ Grounded answers</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

# 6. Session State Init
if "messages" not in st.session_state:
    st.session_state.messages = []
if "eval_log" not in st.session_state:
    st.session_state.eval_log = []

# 7. Chat History Rendering
for i, m in enumerate(st.session_state.messages):
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m["role"] == "assistant" and "meta" in m:
            meta = m["meta"]
            cols = st.columns([1, 1, 1, 3])
            cols[0].markdown(f"<span style='font-size:11px; color:#6C757D;'>⏱ {meta['latency']:.2f}s</span>", unsafe_allow_html=True)
            color = "#3B6D11" if meta['relevance'] >= 0.6 else "#854F0B" if meta['relevance'] >= 0.35 else "#A32D2D"
            cols[1].markdown(f"<span style='font-size:11px; color:{color};'>🎯 {meta['relevance']:.0%} relevance</span>", unsafe_allow_html=True)
            
            if meta.get("feedback") is None:
                with cols[2]:
                    f_col1, f_col2 = st.columns(2)
                    if f_col1.button("👍", key=f"hist_up_{i}"):
                        st.session_state.messages[i]["meta"]["feedback"] = "positive"
                        save_feedback(st.session_state.eval_log, i, "positive")
                        st.rerun()
                    if f_col2.button("👎", key=f"hist_dn_{i}"):
                        st.session_state.messages[i]["meta"]["feedback"] = "negative"
                        save_feedback(st.session_state.eval_log, i, "negative")
                        st.rerun()
            else:
                cols[2].markdown(f"<span style='font-size:11px; color:#6C757D;'>{'👍' if meta['feedback'] == 'positive' else '👎'} Rated</span>", unsafe_allow_html=True)

# 8. Core RAG Logic
bot = init_system()

if prompt := st.chat_input("How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner(""):
            t_start = time.perf_counter()
            response = bot.invoke({"input": prompt})
            t_end = time.perf_counter()
            latency = round(t_end - t_start, 3)

            answer = response["answer"]
            source_docs = response.get("context", [])
            relevance_score = compute_contextual_relevance(prompt, source_docs)

            st.markdown(answer)

            # Instant Metrics and Feedback UI
            color = "#3B6D11" if relevance_score >= 0.6 else "#854F0B" if relevance_score >= 0.35 else "#A32D2D"
            m_cols = st.columns([1, 1, 1, 3])
            m_cols[0].markdown(f"<span style='font-size:11px; color:#6C757D;'>⏱ {latency:.2f}s</span>", unsafe_allow_html=True)
            m_cols[1].markdown(f"<span style='font-size:11px; color:{color};'>🎯 {relevance_score:.0%} relevance</span>", unsafe_allow_html=True)
            
            with m_cols[2]:
                fb_1, fb_2 = st.columns(2)
                # Unique keys using current message index
                curr_idx = len(st.session_state.messages)
                if fb_1.button("👍", key=f"new_up_{curr_idx}"):
                    st.session_state.pending_feedback = (curr_idx, "positive")
                if fb_2.button("👎", key=f"new_dn_{curr_idx}"):
                    st.session_state.pending_feedback = (curr_idx, "negative")

            # Final Session State Update
            meta_data = {"latency": latency, "relevance": relevance_score, "feedback": None}
            st.session_state.eval_log.append({"query": prompt, "answer": answer, "latency": latency, "relevance": relevance_score, "feedback": None})
            st.session_state.messages.append({"role": "assistant", "content": answer, "meta": meta_data})
            
            # Handle pending feedback from buttons just pressed
            if st.session_state.get("pending_feedback"):
                idx, val = st.session_state.pending_feedback
                st.session_state.messages[idx]["meta"]["feedback"] = val
                save_feedback(st.session_state.eval_log, idx, val)
                st.session_state.pending_feedback = None
                st.rerun()