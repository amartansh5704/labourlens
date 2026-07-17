# frontend/components/comparison.py
# Side by side jurisdiction comparison

import streamlit as st
import httpx
from frontend.components.sources import render_sources
from shared.constants import JURISDICTIONS, TOPICS

from frontend.config import API_URL


def render_comparison():
    """
    Renders the jurisdiction comparison view.
    Same question answered for two jurisdictions side by side.
    """

    st.markdown("## ⚖️ Compare Jurisdictions")
    st.markdown(
        "Ask the same question across two jurisdictions "
        "and see the differences side by side."
    )
    st.markdown("---")

    # jurisdiction selectors
    col1, col2 = st.columns(2)

    jurisdiction_list = list(JURISDICTIONS.keys())

    with col1:
        j1 = st.selectbox(
            "Jurisdiction 1",
            jurisdiction_list,
            index=0,
            key="compare_j1"
        )

    with col2:
        j2 = st.selectbox(
            "Jurisdiction 2",
            jurisdiction_list,
            index=1,
            key="compare_j2"
        )

    # topic filter
    topic_display = ["All"] + list(TOPICS.values())
    selected_topic = st.selectbox(
        "Topic (optional)",
        topic_display,
        key="compare_topic"
    )

    topic = None
    if selected_topic != "All":
        topic = list(TOPICS.keys())[
            list(TOPICS.values()).index(selected_topic)
        ]

    # question input
    question = st.text_input(
        "Question to compare",
        placeholder="e.g. What is the minimum wage?",
        key="compare_question"
    )

    # compare button
    if st.button(
        "⚖️ Compare",
        type="primary",
        disabled=not question or j1 == j2
    ):
        if j1 == j2:
            st.warning("Please select two different jurisdictions")
            return

        with st.spinner("Comparing jurisdictions..."):
            try:
                response = httpx.post(
                    f"{API_URL}/compare",
                    json={
                        "question": question,
                        "jurisdiction1": j1,
                        "jurisdiction2": j2,
                        "topic": topic,
                    },
                    timeout=90.0,
                )
                response.raise_for_status()
                result = response.json()

                _render_comparison_result(result, j1, j2)

            except httpx.ConnectError:
                st.error(
                    "Cannot connect to API. "
                    "Start with: uvicorn api.main:app --reload"
                )
            except Exception as e:
                st.error(f"Comparison failed: {e}")

    if j1 == j2:
        st.warning("⚠️ Select two different jurisdictions")


def _render_comparison_result(result: dict, j1: str, j2: str):
    """Render comparison results side by side"""

    st.markdown("---")
    st.markdown(f"**Question:** {result.get('question', '')}")
    st.markdown("---")

    comparison = result.get("comparison", {})

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"### 📍 {j1}")
        j1_data = comparison.get(j1, {})

        if j1_data.get("has_results"):
            st.markdown(j1_data.get("answer", ""))
            sources = j1_data.get("sources", [])
            if sources:
                render_sources(sources)
        else:
            st.warning(f"No data found for {j1}")

    with col2:
        st.markdown(f"### 📍 {j2}")
        j2_data = comparison.get(j2, {})

        if j2_data.get("has_results"):
            st.markdown(j2_data.get("answer", ""))
            sources = j2_data.get("sources", [])
            if sources:
                render_sources(sources)
        else:
            st.warning(f"No data found for {j2}")

    st.markdown("---")
    st.caption(
        "⚠️ For informational purposes only. Not legal advice."
    )