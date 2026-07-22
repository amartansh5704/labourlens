# streamlit_app.py
import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("torch").setLevel(logging.ERROR)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


import streamlit as st

# Load secrets into environment variables
# Streamlit Cloud uses st.secrets instead of .env
def load_secrets():
    """Load Streamlit secrets into os.environ"""
    try:
        for key, value in st.secrets.items():
            os.environ[key] = str(value)
    except Exception:
        # local development - use .env file
        from dotenv import load_dotenv
        load_dotenv()

load_secrets()

# NOW import pipeline (needs env vars loaded first)
from api.rag.pipeline import LegalRAGPipeline
from shared.constants import JURISDICTIONS, TOPICS

# ─────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LaborLens - Indian Employment Law",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.85rem;
        color: #856404;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.title("⚖️ LaborLens")
st.caption(
    "Indian Employment Law Compliance Assistant | "
    "Powered by RAG + Groq Llama 3.3 70B"
)

st.markdown("""
<div class="disclaimer">
⚠️ <strong>Disclaimer:</strong>
LaborLens provides information for educational purposes only.
This is NOT legal advice. Always consult a qualified lawyer.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# LOAD PIPELINE (cached so model loads only once)
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading AI model...")
def load_pipeline():
    return LegalRAGPipeline()

pipeline = load_pipeline()

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Filters")
    st.markdown("---")

    jurisdiction = st.selectbox(
        "🏛️ Jurisdiction",
        ["All"] + list(JURISDICTIONS.keys()),
        help="Filter by Indian state or Central laws"
    )

    topic_names = list(TOPICS.values())
    topic_keys = list(TOPICS.keys())
    selected_topic_name = st.selectbox(
        "📋 Topic",
        ["All"] + topic_names,
        help="Filter by employment law topic"
    )

    topic = None
    if selected_topic_name != "All":
        topic = topic_keys[topic_names.index(selected_topic_name)]

    top_k = st.slider("📊 Sources to fetch", 1, 10, 5)

    st.markdown("---")
    st.markdown("### 💡 Try asking")
    examples = [
        "What is minimum wage in Delhi?",
        "What is EPF contribution rate?",
        "How many hours can workers work per day?",
        "What are maternity leave rights?",
        "Rights of contract workers?",
        "What is ESI contribution rate?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["example_question"] = ex

    st.markdown("---")
    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.rerun()

# ─────────────────────────────────────────────────────────
# CHAT INTERFACE
# ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []

# display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander(
                f"📚 {len(msg['sources'])} Sources"
            ):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(
                        f"**{i}. {src.get('law_name', 'Source')}**"
                    )
                    st.caption(
                        f"🏛️ {src.get('jurisdiction', '')} | "
                        f"📅 {src.get('effective_date', '')} | "
                        f"🏢 {src.get('agency', '')}"
                    )
                    text = src.get("text", "")
                    if text:
                        st.info(
                            text[:400] + "..."
                            if len(text) > 400
                            else text
                        )
                    url = src.get("source_url", "")
                    if url and url.startswith("http"):
                        st.markdown(f"[🔗 Source]({url})")
                    if i < len(msg["sources"]):
                        st.divider()

# handle example question buttons
question = None
if "example_question" in st.session_state:
    question = st.session_state.pop("example_question")
else:
    question = st.chat_input(
        "Ask a compliance question... "
        "e.g. What is minimum wage in Delhi?"
    )

if question:
    # show user message
    with st.chat_message("user"):
        st.markdown(question)

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    # get answer
    with st.chat_message("assistant"):
        with st.spinner("🔍 Searching legal documents..."):
            result = pipeline.run(
                question=question,
                jurisdiction=(
                    None if jurisdiction == "All"
                    else jurisdiction
                ),
                topic=topic,
                top_k=top_k
            )

        # show answer source badge
        source_badges = {
            "indexed_documents": "🟢 From indexed legal documents",
            "web_search": "🔵 From live web search",
            "llm_knowledge": "🟡 From AI general knowledge",
        }
        badge = source_badges.get(
            result.get("answer_source", ""), ""
        )
        if badge:
            st.caption(badge)

        st.markdown(result["answer"])

        sources = result.get("sources", [])
        if sources:
            with st.expander(
                f"📚 View {len(sources)} Sources"
            ):
                for i, src in enumerate(sources, 1):
                    st.markdown(
                        f"**{i}. {src.get('law_name', 'Source')}**"
                    )
                    st.caption(
                        f"🏛️ {src.get('jurisdiction', '')} | "
                        f"Score: {src.get('score', 0):.2f}"
                    )
                    text = src.get("text", "")
                    if text:
                        st.info(
                            text[:400] + "..."
                            if len(text) > 400
                            else text
                        )
                    url = src.get("source_url", "")
                    if url and url.startswith("http"):
                        st.markdown(f"[🔗 Source]({url})")
                    if i < len(sources):
                        st.divider()

        if result.get("is_low_confidence"):
            st.warning(
                "⚠️ Low confidence - please verify "
                "with official sources"
            )

        st.caption(
            "⚠️ Not legal advice. "
            "Consult a qualified lawyer."
        )

    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result.get("sources", []),
    })