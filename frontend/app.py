# frontend/app.py
# Main Streamlit application entry point

import streamlit as st
import sys
import os

# add project root to path
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from frontend.components.filters import render_filters
from frontend.components.chat import render_chat
from frontend.components.comparison import render_comparison
from frontend.components.document_browser import render_document_browser
from frontend.components.stats import render_stats

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# Must be first Streamlit command
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LaborLens - Indian Employment Law",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main header */
    .main-header {
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a4f 100%);
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }

    /* Chat messages */
    .stChatMessage {
        border-radius: 10px;
        margin-bottom: 0.5rem;
    }

    /* Metric cards */
    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e0e0e0;
    }

    /* Info boxes */
    .stInfo {
        border-radius: 8px;
    }

    /* Source cards */
    .stExpander {
        border-radius: 8px;
        border: 1px solid #e0e0e0;
    }

    /* Disclaimer */
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.85rem;
        color: #856404;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>⚖️ LaborLens</h1>
    <p>Indian Employment Law Compliance Assistant</p>
    <p style="font-size:0.85rem; opacity:0.8;">
        Powered by RAG • Groq Llama 3.3 70B • Real Legal Documents
    </p>
</div>
""", unsafe_allow_html=True)

# disclaimer
st.markdown("""
<div class="disclaimer">
    ⚠️ <strong>Disclaimer:</strong>
    LaborLens provides information for educational purposes only.
    This is NOT legal advice.
    Laws change frequently.
    Always consult a qualified employment lawyer for legal matters.
</div>
""", unsafe_allow_html=True)

st.markdown("")

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ⚖️ LaborLens")
    st.markdown("*Indian Employment Law RAG*")
    st.markdown("---")

    # navigation
    st.markdown("### 📌 Navigation")
    page = st.radio(
        "Go to",
        [
            "💬 Chat",
            "⚖️ Compare Jurisdictions",
            "📂 Document Browser",
            "📊 Statistics",
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # filters (only show for chat page)
    if page == "💬 Chat":
        filters = render_filters()
    else:
        filters = {
            "jurisdiction": None,
            "topic": None,
            "top_k": 5,
            "jurisdiction_display": "All",
            "topic_display": "All",
        }

    st.markdown("---")

    # clear chat button
    if page == "💬 Chat":
        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()

    # quick tips
    st.markdown("### 💡 Example Questions")
    tips = [
        "What is minimum wage in Delhi?",
        "What are overtime rules?",
        "What is EPF contribution rate?",
        "What are maternity leave rights?",
        "Rights of contract workers?",
    ]
    for tip in tips:
        st.markdown(f"• *{tip}*")

# ─────────────────────────────────────────────────────────
# MAIN CONTENT
# ─────────────────────────────────────────────────────────
if page == "💬 Chat":
    render_chat(
        jurisdiction=filters["jurisdiction"],
        topic=filters["topic"],
        top_k=filters["top_k"],
        jurisdiction_display=filters["jurisdiction_display"],
        topic_display=filters["topic_display"],
    )

elif page == "⚖️ Compare Jurisdictions":
    render_comparison()

elif page == "📂 Document Browser":
    render_document_browser()

elif page == "📊 Statistics":
    render_stats()