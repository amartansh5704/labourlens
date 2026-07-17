# frontend/components/document_browser.py
# Browse all indexed documents

import streamlit as st
import httpx
from shared.constants import JURISDICTIONS, TOPICS

from frontend.config import API_URL


def render_document_browser():
    """Browse all scraped and indexed documents"""

    st.markdown("## 📂 Document Browser")
    st.markdown("Browse all indexed legal documents.")
    st.markdown("---")

    # filters
    col1, col2, col3 = st.columns(3)

    with col1:
        j_options = ["All"] + list(JURISDICTIONS.keys())
        filter_j = st.selectbox(
            "Filter by Jurisdiction",
            j_options,
            key="browser_j"
        )

    with col2:
        t_options = ["All"] + list(TOPICS.values())
        filter_t_display = st.selectbox(
            "Filter by Topic",
            t_options,
            key="browser_t"
        )
        filter_t = None
        if filter_t_display != "All":
            filter_t = list(TOPICS.keys())[
                list(TOPICS.values()).index(filter_t_display)
            ]

    with col3:
        filter_indexed = st.selectbox(
            "Index Status",
            ["All", "Indexed", "Not Indexed"],
            key="browser_indexed"
        )

    # build query params
    params = {"page": 1, "page_size": 20}
    if filter_j != "All":
        params["jurisdiction"] = filter_j
    if filter_t:
        params["topic"] = filter_t
    if filter_indexed == "Indexed":
        params["is_indexed"] = True
    elif filter_indexed == "Not Indexed":
        params["is_indexed"] = False

    # fetch documents
    try:
        response = httpx.get(
            f"{API_URL}/documents",
            params=params,
            timeout=15.0
        )
        response.raise_for_status()
        data = response.json()

        documents = data.get("documents", [])
        total = data.get("total", 0)

        st.markdown(f"**{total} documents found**")
        st.markdown("---")

        if not documents:
            st.info(
                "No documents found. "
                "Run the scraper to add documents:\n"
                "`python scraper/ingest_documents.py`"
            )
            return

        # display documents as cards
        for doc in documents:
            _render_document_card(doc)

    except httpx.ConnectError:
        st.error("Cannot connect to API server.")
    except Exception as e:
        st.error(f"Failed to load documents: {e}")


def _render_document_card(doc: dict):
    """Render one document as an expandable card"""

    title = doc.get("title") or doc.get("law_name") or "Untitled"
    jurisdiction = doc.get("jurisdiction", "")
    topic = doc.get("topic", "")
    is_indexed = doc.get("is_indexed", False)
    chunk_count = doc.get("chunk_count", 0)
    file_type = doc.get("file_type", "")

    # status indicator
    status = "✅ Indexed" if is_indexed else "⏳ Not indexed"

    with st.expander(
        f"{status} | {jurisdiction} | {title[:60]}"
    ):
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Jurisdiction", jurisdiction)
        with col2:
            topic_display = TOPICS.get(topic, topic)
            st.metric("Topic", topic_display[:15])
        with col3:
            st.metric("Chunks", chunk_count)
        with col4:
            st.metric("Type", file_type.upper())

        # law name
        law_name = doc.get("law_name", "")
        if law_name:
            st.markdown(f"**Law:** {law_name}")

        # agency
        agency = doc.get("agency", "")
        if agency:
            st.markdown(f"**Agency:** {agency}")

        # scraped date
        scraped_at = doc.get("scraped_at", "")
        if scraped_at:
            st.markdown(f"**Scraped:** {scraped_at[:10]}")

        # text preview
        preview = doc.get("text_preview", "")
        if preview:
            st.markdown("**Content preview:**")
            st.text(preview[:300])

        # source URL
        url = doc.get("url", "")
        if url and url.startswith("http"):
            st.markdown(f"[🔗 Source]({url})")