# frontend/components/chat.py
# Main chat interface component

import streamlit as st
import httpx
from frontend.components.sources import render_sources
from typing import Optional

from frontend.config import API_URL


def render_chat(
    jurisdiction: Optional[str],
    topic: Optional[str],
    top_k: int,
    jurisdiction_display: str,
    topic_display: str,
):
    """
    Renders the main chat interface.
    Handles message history and API calls.
    """

    # init chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # display existing messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                render_sources(message["sources"])
            if message.get("is_low_confidence"):
                st.warning(
                    "⚠️ Low confidence answer - "
                    "retrieved documents may not directly "
                    "answer this question"
                )

    # show filter context
    filter_text = []
    if jurisdiction_display != "All":
        filter_text.append(f"📍 {jurisdiction_display}")
    if topic_display != "All":
        filter_text.append(f"📋 {topic_display}")

    if filter_text:
        st.caption(
            "Active filters: " + " | ".join(filter_text)
        )

    # chat input
    question = st.chat_input(
        "Ask a compliance question... "
        "e.g. What is the minimum wage in Delhi?"
    )

    if question:
        # display user message
        with st.chat_message("user"):
            st.markdown(question)

        st.session_state.messages.append({
            "role": "user",
            "content": question,
        })

        # get answer from API
        with st.chat_message("assistant"):
            with st.spinner("🔍 Searching legal documents..."):
                result = _call_api(
                    question=question,
                    jurisdiction=jurisdiction,
                    topic=topic,
                    top_k=top_k,
                )

            if result:
                _render_answer(result)

                # save to history
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result.get("answer", ""),
                    "sources": result.get("sources", []),
                    "is_low_confidence": result.get(
                        "is_low_confidence", False
                    ),
                })
            else:
                error_msg = (
                    "❌ Could not connect to API. "
                    "Make sure the API is running:\n"
                    "`uvicorn api.main:app --reload --port 8000`"
                )
                st.error(error_msg)


def _call_api(
    question: str,
    jurisdiction: Optional[str],
    topic: Optional[str],
    top_k: int,
) -> Optional[dict]:
    """Call the FastAPI backend"""
    try:
        response = httpx.post(
            f"{API_URL}/ask",
            json={
                "question": question,
                "jurisdiction": jurisdiction,
                "topic": topic,
                "top_k": top_k,
            },
            timeout=60.0,
        )
        response.raise_for_status()
        return response.json()

    except httpx.ConnectError:
        st.error(
            "❌ Cannot connect to API server.\n\n"
            "Start it with:\n"
            "```\nuvicorn api.main:app --reload --port 8000\n```"
        )
        return None

    except httpx.TimeoutException:
        st.error("⏱️ Request timed out. Try again.")
        return None

    except Exception as e:
        st.error(f"❌ API error: {str(e)}")
        return None


def _render_answer(result: dict):
    """Render answer with source indicator"""

    answer = result.get("answer", "")
    sources = result.get("sources", [])
    has_results = result.get("has_results", False)
    is_low_confidence = result.get("is_low_confidence", False)
    answer_source = result.get("answer_source", "unknown")

    # show where answer came from
    source_badges = {
        "indexed_documents": "🟢 From our indexed legal documents",
        "web_search":        "🔵 From live web search",
        "llm_knowledge":     "🟡 From AI general knowledge",
        "error":             "🔴 Error occurred",
    }

    badge = source_badges.get(answer_source, "")
    if badge:
        st.caption(badge)

    # render answer
    st.markdown(answer)

    # low confidence warning
    if is_low_confidence and answer_source == "llm_knowledge":
        st.warning(
            "⚠️ This answer is from AI general knowledge, "
            "not from our indexed documents. "
            "Please verify with official sources."
        )

    # web search warning
    if answer_source == "web_search":
        st.info(
            "🌐 Answer sourced from live web search. "
            "Information may not be from official sources."
        )

    # render sources
    if sources:
        render_sources(sources)

    # disclaimer
    st.caption(
        "⚠️ For informational purposes only. "
        "Not legal advice. Consult a qualified lawyer."
    )