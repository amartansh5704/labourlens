# frontend/components/stats.py
# System statistics dashboard

import streamlit as st
import httpx

API_URL = "http://localhost:8000/api"


def render_stats():
    """Render system statistics dashboard"""

    st.markdown("## 📊 System Statistics")
    st.markdown("---")

    try:
        # fetch stats
        response = httpx.get(
            f"{API_URL}/stats",
            timeout=15.0
        )
        response.raise_for_status()
        stats = response.json()

        # top metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "📄 Total Documents",
                stats.get("total_documents", 0)
            )
        with col2:
            st.metric(
                "✅ Indexed",
                stats.get("indexed_documents", 0)
            )
        with col3:
            st.metric(
                "🔢 Total Chunks",
                stats.get("total_chunks", 0)
            )
        with col4:
            st.metric(
                "🧠 Qdrant Vectors",
                stats.get("qdrant_vectors", 0)
            )

        st.markdown("---")

        # by jurisdiction
        col1, col2 = st.columns(2)

        with col1:
            st.markdown("### By Jurisdiction")
            by_j = stats.get("by_jurisdiction", {})
            if by_j:
                for jurisdiction, count in by_j.items():
                    bar_width = min(count * 3, 30)
                    bar = "█" * bar_width
                    st.markdown(
                        f"`{jurisdiction:<15}` "
                        f"**{count}** docs {bar}"
                    )
            else:
                st.info("No documents yet")

        with col2:
            st.markdown("### By Topic")
            by_t = stats.get("by_topic", {})
            if by_t:
                from shared.constants import TOPICS
                for topic_key, count in by_t.items():
                    topic_name = TOPICS.get(
                        topic_key, topic_key
                    )
                    bar_width = min(count * 3, 30)
                    bar = "█" * bar_width
                    st.markdown(
                        f"`{topic_name[:20]:<20}` "
                        f"**{count}** {bar}"
                    )
            else:
                st.info("No documents yet")

        st.markdown("---")

        # health check
        st.markdown("### 🔧 System Health")
        health_response = httpx.get(
            f"{API_URL}/health",
            timeout=10.0
        )
        health = health_response.json()

        h_col1, h_col2, h_col3, h_col4 = st.columns(4)

        def status_indicator(status):
            return "🟢" if status == "ok" else "🔴"

        with h_col1:
            st.markdown(
                f"{status_indicator(health.get('database', 'error'))} "
                f"**Database**\n\n{health.get('database', 'error')}"
            )
        with h_col2:
            st.markdown(
                f"{status_indicator(health.get('qdrant', 'error'))} "
                f"**Qdrant**\n\n{health.get('qdrant', 'error')}"
            )
        with h_col3:
            st.markdown(
                f"{status_indicator(health.get('groq', 'error'))} "
                f"**Groq API**\n\n{health.get('groq', 'error')}"
            )
        with h_col4:
            st.markdown(
                f"**Version**\n\n{health.get('version', '1.0.0')}"
            )

    except httpx.ConnectError:
        st.error(
            "Cannot connect to API server.\n\n"
            "Start it: `uvicorn api.main:app --reload --port 8000`"
        )
    except Exception as e:
        st.error(f"Failed to load stats: {e}")

    # refresh button
    st.markdown("---")
    if st.button("🔄 Refresh Stats"):
        st.rerun()