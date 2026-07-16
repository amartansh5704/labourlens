# frontend/components/filters.py
# Sidebar filters component

import streamlit as st
from shared.constants import JURISDICTIONS, TOPICS


def render_filters() -> dict:
    """
    Renders sidebar filters.
    Returns dict with selected values.
    """

    st.sidebar.markdown("## ⚙️ Filters")
    st.sidebar.markdown("---")

    # jurisdiction filter
    jurisdiction_options = ["All"] + list(JURISDICTIONS.keys())
    jurisdiction = st.sidebar.selectbox(
        "🏛️ Jurisdiction",
        jurisdiction_options,
        index=0,
        help="Filter by Indian state or Central laws"
    )

    # topic filter
    topic_display = ["All"] + list(TOPICS.values())
    topic_keys = [None] + list(TOPICS.keys())

    selected_topic_label = st.sidebar.selectbox(
        "📋 Topic",
        topic_display,
        index=0,
        help="Filter by employment law topic"
    )

    # convert display name back to key
    topic = None
    if selected_topic_label != "All":
        topic = list(TOPICS.keys())[
            list(TOPICS.values()).index(selected_topic_label)
        ]

    # results count slider
    top_k = st.sidebar.slider(
        "📊 Results to fetch",
        min_value=1,
        max_value=10,
        value=5,
        help="How many source passages to retrieve"
    )

    st.sidebar.markdown("---")

    # info box
    if jurisdiction != "All":
        st.sidebar.info(
            f"Searching: **{JURISDICTIONS.get(jurisdiction, jurisdiction)}**"
        )

    return {
        "jurisdiction": None if jurisdiction == "All" else jurisdiction,
        "topic": topic,
        "top_k": top_k,
        "jurisdiction_display": jurisdiction,
        "topic_display": selected_topic_label,
    }