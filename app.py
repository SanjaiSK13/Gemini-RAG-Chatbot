import streamlit as st
import os
from src.data_loader import load_and_split_data
from src.embedding_engine import create_vector_store, load_vector_store
from src.rag_chain import get_rag_chain
from src.config import LLM_MODEL

# 1. Page Identity 
st.set_page_config(page_title="Sanjai Support AI", page_icon="🛡️", layout="wide")

# 2. Professional CSS 
st.markdown("""
    <style>
    .stApp { background-color: #F8F9FA; }
    .main-header { color: #1E3A8A; font-size: 2.2rem; font-weight: 700; margin-bottom: 0px; }
    .status-text { font-size: 0.8rem; color: #6C757D; }
    /* Chat Bubble Styling */
    .stChatMessage { border-radius: 12px; border: 1px solid #E9ECEF; box-shadow: 0 2px 4px rgba(0,0,0,0.02); }
    </style>
    """, unsafe_allow_html=True)

# 3. High-Speed Caching
@st.cache_resource(show_spinner=False)
def init_system():
    if not os.path.exists("vector_store"): os.makedirs("vector_store")
    if not os.path.exists("vector_store/faiss_index"):
        chunks = load_and_split_data()
        vs = create_vector_store(chunks)
    else:
        vs = load_vector_store()
    return get_rag_chain(vs)

# 4. Sidebar: Metadata Display
with st.sidebar:
    st.markdown("### ⚙️ System Profile")
    st.success(f"**Model:** {LLM_MODEL}")
    st.info("**Database:** FAISS Vector Index")
    st.markdown("---")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

# 5. UI Header
st.markdown('<p class="main-header">🛡️ Intelligence Support Portal</p>', unsafe_allow_html=True)
st.markdown('<p class="status-text">Grounded RAG Pipeline | Security Verified</p>', unsafe_allow_html=True)

# 6. Chat History Rendering 
if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# 7. Core RAG Logic
bot = init_system()

if prompt := st.chat_input("How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        with st.spinner(""): # Minimal spinner for speed
            # DIRECT INVOKE: No typewriter, no delays
            response = bot.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})