# frontend/components/sources.py
# Renders source documents panel

import streamlit as st
from typing import List, Dict


def render_sources(sources: List[Dict]):
    """
    Renders expandable source documents panel.
    Shows exact quotes, law names, source URLs.
    """

    if not sources:
        return

    with st.expander(
        f"📚 View {len(sources)} Source{'s' if len(sources) > 1 else ''}",
        expanded=False
    ):
        for i, source in enumerate(sources, 1):

            # source header
            col1, col2 = st.columns([3, 1])

            with col1:
                law_name = source.get("law_name") or "Unknown Law"
                st.markdown(f"**{i}. {law_name}**")

            with col2:
                score = source.get("score", 0)
                score_color = (
                    "🟢" if score > 0.7 else
                    "🟡" if score > 0.5 else
                    "🟠"
                )
                st.markdown(
                    f"{score_color} `{score:.2f}`",
                    help="Relevance score (higher is better)"
                )

            # metadata row
            meta_cols = st.columns(3)

            with meta_cols[0]:
                jurisdiction = source.get("jurisdiction", "")
                if jurisdiction:
                    st.markdown(f"🏛️ **{jurisdiction}**")

            with meta_cols[1]:
                effective_date = source.get("effective_date", "")
                if effective_date:
                    st.markdown(f"📅 {effective_date}")

            with meta_cols[2]:
                agency = source.get("agency", "")
                if agency:
                    st.markdown(
                        f"🏢 {agency[:30]}..."
                        if len(agency) > 30
                        else f"🏢 {agency}"
                    )

            # exact quote from law
            text = source.get("text", "")
            if text:
                st.markdown("**Relevant passage:**")
                st.info(
                    text[:600] + "..."
                    if len(text) > 600
                    else text
                )

            # source link
            source_url = source.get("source_url", "")
            if source_url and source_url.startswith("http"):
                st.markdown(
                    f"[🔗 View Original Source]({source_url})"
                )

            if i < len(sources):
                st.divider()