# streamlit_app.py
# Full LaborLens for Streamlit Cloud deployment
# All helper functions defined before use

import warnings
warnings.filterwarnings("ignore")

import logging
logging.getLogger("torch").setLevel(logging.ERROR)

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import streamlit as st

# ─────────────────────────────────────────────────────────
# PAGE CONFIG - must be first streamlit command
# ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="LaborLens - Indian Employment Law",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────
# LOAD SECRETS INTO ENVIRONMENT
# ─────────────────────────────────────────────────────────
def load_secrets():
    try:
        for key, value in st.secrets.items():
            os.environ[key] = str(value)
    except Exception:
        from dotenv import load_dotenv
        load_dotenv()

load_secrets()

# ─────────────────────────────────────────────────────────
# INITIALIZE DATABASE (create tables if missing)
# ─────────────────────────────────────────────────────────
from database.connection import init_db
init_db()

# ─────────────────────────────────────────────────────────
# IMPORTS
# ─────────────────────────────────────────────────────────
from api.rag.pipeline import LegalRAGPipeline
from shared.constants import JURISDICTIONS, TOPICS
from database.connection import get_db_session, get_db_stats
from database.models import Document
from qdrant_client import QdrantClient

# ─────────────────────────────────────────────────────────
# HELPER FUNCTIONS - defined before any page code
# ─────────────────────────────────────────────────────────

def _get_qdrant_count() -> int:
    """Get total vectors in Qdrant"""
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "")
        qdrant_key = os.getenv("QDRANT_API_KEY", "")
        qdrant_col = os.getenv(
            "QDRANT_COLLECTION",
            "employment_law_india"
        )
        if qdrant_key and "cloud.qdrant.io" in qdrant_host:
            qclient = QdrantClient(
                url=f"https://{qdrant_host}",
                api_key=qdrant_key,
                check_compatibility=False,
            )
        else:
            qclient = QdrantClient(
                host=qdrant_host,
                port=int(os.getenv("QDRANT_PORT", 6333))
            )
        return qclient.count(
            collection_name=qdrant_col,
            exact=True
        ).count
    except Exception:
        return 0


def _render_sources(sources: list):
    """Render source documents in expandable panel"""
    if not sources:
        return

    with st.expander(
        f"📚 View {len(sources)} "
        f"Source{'s' if len(sources) > 1 else ''}"
    ):
        for i, source in enumerate(sources, 1):
            law_name = (
                source.get("law_name") or "Unknown Law"
            )
            score = source.get("score", 0)

            col1, col2 = st.columns([3, 1])
            with col1:
                st.markdown(f"**{i}. {law_name}**")
            with col2:
                score_emoji = (
                    "🟢" if score > 0.7 else
                    "🟡" if score > 0.5 else
                    "🟠"
                )
                st.markdown(f"{score_emoji} `{score:.2f}`")

            meta_cols = st.columns(3)
            with meta_cols[0]:
                if source.get("jurisdiction"):
                    st.markdown(
                        f"🏛️ **{source['jurisdiction']}**"
                    )
            with meta_cols[1]:
                if source.get("effective_date"):
                    st.markdown(
                        f"📅 {source['effective_date']}"
                    )
            with meta_cols[2]:
                agency = source.get("agency", "")
                if agency:
                    display_agency = (
                        f"{agency[:25]}..."
                        if len(agency) > 25
                        else agency
                    )
                    st.markdown(f"🏢 {display_agency}")

            text = source.get("text", "")
            if text:
                st.markdown("**Relevant passage:**")
                display_text = (
                    text[:500] + "..."
                    if len(text) > 500
                    else text
                )
                st.info(display_text)

            url = source.get("source_url", "")
            if url and url.startswith("http"):
                st.markdown(
                    f"[🔗 View Original Source]({url})"
                )

            if i < len(sources):
                st.divider()

# Add this function next to _get_qdrant_count and _render_sources

def _get_documents_from_qdrant(
    jurisdiction_filter: str = "All",
    topic_filter: str = None,
) -> list:
    """
    Get unique documents from Qdrant payloads.
    Since SQLite is empty on cloud, we extract
    document metadata from Qdrant chunk payloads.
    """
    try:
        qdrant_host = os.getenv("QDRANT_HOST", "")
        qdrant_key = os.getenv("QDRANT_API_KEY", "")
        qdrant_col = os.getenv(
            "QDRANT_COLLECTION",
            "employment_law_india"
        )

        if qdrant_key and "cloud.qdrant.io" in qdrant_host:
            qclient = QdrantClient(
                url=f"https://{qdrant_host}",
                api_key=qdrant_key,
                check_compatibility=False,
            )
        else:
            qclient = QdrantClient(
                host=qdrant_host,
                port=int(os.getenv("QDRANT_PORT", 6333))
            )

        # build filter
        from qdrant_client.models import (
            Filter, FieldCondition, MatchValue
        )

        conditions = []
        if (
            jurisdiction_filter and
            jurisdiction_filter != "All"
        ):
            conditions.append(
                FieldCondition(
                    key="jurisdiction",
                    match=MatchValue(
                        value=jurisdiction_filter
                    )
                )
            )
        if topic_filter and topic_filter != "All":
            conditions.append(
                FieldCondition(
                    key="topic",
                    match=MatchValue(value=topic_filter)
                )
            )

        scroll_filter = (
            Filter(must=conditions) if conditions
            else None
        )

        # scroll through all points to get payloads
        all_points = []
        offset = None

        while True:
            results, next_offset = qclient.scroll(
                collection_name=qdrant_col,
                scroll_filter=scroll_filter,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            all_points.extend(results)

            if next_offset is None:
                break
            offset = next_offset

        # extract unique documents from chunk payloads
        seen_docs = {}

        for point in all_points:
            payload = point.payload or {}
            doc_id = payload.get(
                "document_id",
                payload.get("source_url", "unknown")
            )

            if doc_id not in seen_docs:
                seen_docs[doc_id] = {
                    "id": doc_id,
                    "title": payload.get("title", ""),
                    "law_name": payload.get(
                        "law_name", "Unknown"
                    ),
                    "jurisdiction": payload.get(
                        "jurisdiction", ""
                    ),
                    "topic": payload.get("topic", ""),
                    "agency": payload.get("agency", ""),
                    "source_url": payload.get(
                        "source_url", ""
                    ),
                    "effective_date": payload.get(
                        "effective_date", ""
                    ),
                    "document_type": payload.get(
                        "document_type", ""
                    ),
                    "file_type": payload.get(
                        "file_type", ""
                    ),
                    "chunk_count": 0,
                    "sample_text": payload.get(
                        "text", ""
                    )[:300],
                }

            # count chunks per document
            seen_docs[doc_id]["chunk_count"] += 1

        # sort by jurisdiction then law name
        documents = sorted(
            seen_docs.values(),
            key=lambda d: (
                d["jurisdiction"],
                d["law_name"]
            )
        )

        return documents

    except Exception as e:
        st.error(f"Failed to load from Qdrant: {e}")
        return []


# ─────────────────────────────────────────────────────────
# LOAD PIPELINE - cached so model loads only once
# ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="⏳ Loading AI model...")
def load_pipeline():
    return LegalRAGPipeline()


# ─────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    .main-header {
        background: linear-gradient(
            135deg, #1e3a5f 0%, #2d6a4f 100%
        );
        padding: 1.5rem 2rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        color: white;
    }

    .disclaimer {
        background-color: #fff3cd;
        border: 1px solid #ffc107;
        border-radius: 8px;
        padding: 0.75rem;
        font-size: 0.85rem;
        color: #856404;
        margin-bottom: 1rem;
    }

    [data-testid="metric-container"] {
        background-color: #f0f2f6;
        border-radius: 8px;
        padding: 1rem;
        border: 1px solid #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1 style="margin:0; color:white;">⚖️ LaborLens</h1>
    <p style="margin:0; color:white; opacity:0.9;">
        Indian Employment Law Compliance Assistant
    </p>
    <p style="margin:0; font-size:0.85rem; color:white; opacity:0.7;">
        Powered by RAG • Groq Llama 3.3 70B • Real Legal Documents
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="disclaimer">
⚠️ <strong>Disclaimer:</strong>
LaborLens provides information for educational purposes only.
This is NOT legal advice. Laws change frequently.
Always consult a qualified employment lawyer.
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("# ⚖️ LaborLens")
    st.markdown("*Indian Employment Law RAG*")
    st.markdown("---")

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

    # filters only for chat page
    if page == "💬 Chat":
        st.markdown("## ⚙️ Filters")

        jurisdiction = st.selectbox(
            "🏛️ Jurisdiction",
            ["All"] + list(JURISDICTIONS.keys()),
            help="Filter by state or Central laws"
        )

        topic_display_list = (
            ["All"] + list(TOPICS.values())
        )
        selected_topic_label = st.selectbox(
            "📋 Topic",
            topic_display_list,
            help="Filter by employment law topic"
        )

        topic = None
        if selected_topic_label != "All":
            topic = list(TOPICS.keys())[
                list(TOPICS.values()).index(
                    selected_topic_label
                )
            ]

        top_k = st.slider(
            "📊 Results to fetch", 1, 10, 5
        )

        st.markdown("---")

        if st.button("🗑️ Clear Chat History"):
            st.session_state.messages = []
            st.rerun()
    else:
        jurisdiction = "All"
        selected_topic_label = "All"
        topic = None
        top_k = 5

    st.markdown("---")

    st.markdown("### 💡 Example Questions")
    examples = [
        "What is minimum wage in Delhi?",
        "What is EPF contribution rate?",
        "How many hours can workers work?",
        "What are maternity leave rights?",
        "Rights of contract workers?",
        "What is ESI contribution rate?",
    ]
    for ex in examples:
        if st.button(ex, use_container_width=True):
            st.session_state["example_question"] = ex


# ═════════════════════════════════════════════════════════
# PAGE 1: CHAT
# ═════════════════════════════════════════════════════════
if page == "💬 Chat":

    pipeline = load_pipeline()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # show active filters
    filter_parts = []
    if jurisdiction != "All":
        filter_parts.append(f"📍 {jurisdiction}")
    if selected_topic_label != "All":
        filter_parts.append(f"📋 {selected_topic_label}")
    if filter_parts:
        st.caption(
            "Active filters: " + " | ".join(filter_parts)
        )

    # display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                _render_sources(message["sources"])

    # handle example question from sidebar
    question = None
    if "example_question" in st.session_state:
        question = st.session_state.pop("example_question")
    else:
        question = st.chat_input(
            "Ask a compliance question..."
        )

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("assistant"):
            with st.spinner(
                "🔍 Searching legal documents..."
            ):
                result = pipeline.run(
                    question=question,
                    jurisdiction=(
                        None if jurisdiction == "All"
                        else jurisdiction
                    ),
                    topic=topic,
                    top_k=top_k,
                )

            # answer source badge
            badges = {
                "indexed_documents":
                    "🟢 From indexed legal documents",
                "web_search":
                    "🔵 From live web search",
                "llm_knowledge":
                    "🟡 From AI general knowledge",
                "all_sources":
                    "🟢 From documents + 🌐 web + 🧠 AI knowledge",
            }
            badge = badges.get(
                result.get("answer_source", ""), ""
            )
            if badge:
                st.caption(badge)

            st.markdown(result["answer"])

            sources = result.get("sources", [])
        user_wants_sources = any(
            word in question.lower()
            for word in [
                "source", "sources", "reference",
                "where", "citation", "proof",
                "evidence", "link", "url",
                "which law", "what law",
                "show me", "tell me more",
                "details", "detail",
            ]
        )

        if sources and user_wants_sources:
            _render_sources(sources)
        elif sources:
            # show subtle hint instead of full sources
            st.caption(
                f"📚 {len(sources)} sources available. "
                f"Ask 'show sources' to see them."
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


# ═════════════════════════════════════════════════════════
# PAGE 2: COMPARE JURISDICTIONS
# ═════════════════════════════════════════════════════════
elif page == "⚖️ Compare Jurisdictions":

    pipeline = load_pipeline()

    st.markdown("## ⚖️ Compare Jurisdictions")
    st.markdown(
        "Ask the same question for two jurisdictions "
        "and compare answers side by side."
    )
    st.markdown("---")

    col1, col2 = st.columns(2)
    j_list = list(JURISDICTIONS.keys())

    with col1:
        j1 = st.selectbox(
            "Jurisdiction 1", j_list,
            index=0, key="cmp_j1"
        )
    with col2:
        j2 = st.selectbox(
            "Jurisdiction 2", j_list,
            index=1, key="cmp_j2"
        )

    t_opts = ["All"] + list(TOPICS.values())
    sel_cmp_topic = st.selectbox(
        "Topic (optional)", t_opts, key="cmp_topic"
    )
    cmp_topic = None
    if sel_cmp_topic != "All":
        cmp_topic = list(TOPICS.keys())[
            list(TOPICS.values()).index(sel_cmp_topic)
        ]

    cmp_question = st.text_input(
        "Question to compare",
        placeholder="e.g. What is the minimum wage?",
        key="cmp_q"
    )

    if st.button(
        "⚖️ Compare", type="primary",
        disabled=(not cmp_question or j1 == j2)
    ):
        with st.spinner("Comparing..."):
            result = pipeline.run_comparison(
                question=cmp_question,
                jurisdiction1=j1,
                jurisdiction2=j2,
                topic=cmp_topic,
            )

        st.markdown("---")
        st.markdown(f"**Question:** {cmp_question}")
        st.markdown("---")

        comparison = result.get("comparison", {})
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"### 📍 {j1}")
            j1d = comparison.get(j1, {})
            if j1d.get("has_results"):
                st.markdown(j1d.get("answer", ""))
                if j1d.get("sources"):
                    _render_sources(j1d["sources"])
            else:
                st.warning(f"No data for {j1}")

        with col2:
            st.markdown(f"### 📍 {j2}")
            j2d = comparison.get(j2, {})
            if j2d.get("has_results"):
                st.markdown(j2d.get("answer", ""))
                if j2d.get("sources"):
                    _render_sources(j2d["sources"])
            else:
                st.warning(f"No data for {j2}")

        st.caption(
            "⚠️ Not legal advice. Consult a lawyer."
        )

    if j1 == j2:
        st.warning("Select two different jurisdictions")


# ═════════════════════════════════════════════════════════
# PAGE 3: DOCUMENT BROWSER
# ═════════════════════════════════════════════════════════
elif page == "📂 Document Browser":

    st.markdown("## 📂 Document Browser")
    st.markdown("Browse all indexed legal documents.")
    st.markdown("---")

    # show total vectors
    vec_count = _get_qdrant_count()
    if vec_count > 0:
        st.success(
            f"✅ **{vec_count} law passages** indexed "
            f"in Qdrant Cloud"
        )

    # filters
    col1, col2 = st.columns(2)

    with col1:
        j_filter = st.selectbox(
            "Filter by Jurisdiction",
            ["All"] + list(JURISDICTIONS.keys()),
            key="browser_j"
        )

    with col2:
        t_opts = ["All"] + list(TOPICS.values())
        t_filter_display = st.selectbox(
            "Filter by Topic",
            t_opts,
            key="browser_t"
        )
        t_filter = None
        if t_filter_display != "All":
            t_filter = list(TOPICS.keys())[
                list(TOPICS.values()).index(
                    t_filter_display
                )
            ]

    # get documents from Qdrant
    with st.spinner("Loading documents from Qdrant..."):
        documents = _get_documents_from_qdrant(
            jurisdiction_filter=j_filter,
            topic_filter=t_filter,
        )

    st.markdown(
        f"**{len(documents)} documents found | "
        f"{vec_count} total chunks**"
    )
    st.markdown("---")

    if not documents:
        st.info(
            "No documents found for this filter. "
            "Try selecting 'All' for both filters."
        )
    else:
        for doc in documents:
            law_name = (
                doc.get("law_name") or
                doc.get("title") or
                "Untitled"
            )
            jurisdiction = doc.get("jurisdiction", "")
            chunk_count = doc.get("chunk_count", 0)
            topic_key = doc.get("topic", "")
            topic_name = TOPICS.get(topic_key, topic_key)

            with st.expander(
                f"✅ {jurisdiction} | "
                f"{law_name[:50]} | "
                f"{chunk_count} chunks"
            ):
                # metrics row
                c1, c2, c3, c4 = st.columns(4)
                with c1:
                    st.metric(
                        "Jurisdiction", jurisdiction
                    )
                with c2:
                    st.metric(
                        "Topic", topic_name[:15]
                    )
                with c3:
                    st.metric("Chunks", chunk_count)
                with c4:
                    file_type = doc.get(
                        "file_type", "unknown"
                    ).upper()
                    st.metric("Type", file_type)

                # details
                if doc.get("law_name"):
                    st.markdown(
                        f"**📜 Law:** {doc['law_name']}"
                    )
                if doc.get("agency"):
                    st.markdown(
                        f"**🏢 Agency:** {doc['agency']}"
                    )
                if doc.get("effective_date"):
                    st.markdown(
                        f"**📅 Effective:** "
                        f"{doc['effective_date']}"
                    )
                if doc.get("document_type"):
                    st.markdown(
                        f"**📋 Type:** "
                        f"{doc['document_type'].title()}"
                    )

                # sample text
                sample = doc.get("sample_text", "")
                if sample:
                    st.markdown("**📝 Content Preview:**")
                    st.info(
                        sample + "..."
                        if len(sample) >= 295
                        else sample
                    )

                # source link
                url = doc.get("source_url", "")
                if url and url.startswith("http"):
                    st.markdown(
                        f"[🔗 View Original Source]({url})"
                    )


# ═════════════════════════════════════════════════════════
# PAGE 4: STATISTICS
# ═════════════════════════════════════════════════════════
elif page == "📊 Statistics":

    st.markdown("## 📊 System Statistics")
    st.markdown("---")

    vec_count = _get_qdrant_count()

    # get documents from Qdrant for stats
    all_docs = _get_documents_from_qdrant()

    # calculate stats from Qdrant documents
    total_docs = len(all_docs)
    total_chunks = sum(
        d.get("chunk_count", 0) for d in all_docs
    )

    by_jurisdiction = {}
    for j in JURISDICTIONS.keys():
        count = len([
            d for d in all_docs
            if d.get("jurisdiction") == j
        ])
        by_jurisdiction[j] = count

    by_topic = {}
    for t in TOPICS.keys():
        count = len([
            d for d in all_docs
            if d.get("topic") == t
        ])
        by_topic[t] = count

    # top metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📄 Documents", total_docs)
    with col2:
        st.metric("✅ All Indexed", total_docs)
    with col3:
        st.metric("🔢 Total Chunks", total_chunks)
    with col4:
        st.metric("🧠 Qdrant Vectors", vec_count)

    st.markdown("---")

    if vec_count > 0:
        st.success(
            f"✅ **{vec_count}** law passages from "
            f"**{total_docs}** documents indexed "
            f"in Qdrant Cloud"
        )

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### 🏛️ By Jurisdiction")
        for j, count in by_jurisdiction.items():
            bar = "█" * min(count * 3, 20)
            st.markdown(
                f"`{j:<15}` **{count}** docs {bar}"
            )

    with col2:
        st.markdown("### 📋 By Topic")
        for t_key, count in by_topic.items():
            t_name = TOPICS.get(t_key, t_key)
            bar = "█" * min(count * 3, 20)
            st.markdown(
                f"`{t_name[:20]:<20}` **{count}** {bar}"
            )

    st.markdown("---")
    if st.button("🔄 Refresh"):
        st.rerun()