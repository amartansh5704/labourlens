# api/rag/pipeline.py
# Smart pipeline with 3 tier fallback:
# Tier 1: Our Qdrant database (best, most trusted)
# Tier 2: Live web search (good for missing data)
# Tier 3: LLM general knowledge (always available)

from api.rag.retriever import LegalRetriever
from api.rag.llm import GroqLLM
from api.rag.web_search import WebSearcher
from api.rag.prompts import (
    build_rag_prompt,
    build_context_string,
    build_no_results_message,
    build_web_search_prompt,
    build_llm_knowledge_prompt,
    SYSTEM_PROMPT,
)
from api.core.config import settings
from shared.constants import DISCLAIMER
from loguru import logger
from typing import Optional, Dict
import os


class LegalRAGPipeline:
    """
    3-tier RAG pipeline:

    TIER 1 - Qdrant (our indexed documents)
    ─────────────────────────────────────────
    Best quality, most trusted
    Answers from real Indian law documents we indexed
    Always tried first

    TIER 2 - Web Search (Tavily)
    ─────────────────────────────────────────
    Live internet search focused on Indian labor law sites
    Used when Qdrant has no relevant documents
    Good for: missing states, recent changes, current rates

    TIER 3 - LLM General Knowledge
    ─────────────────────────────────────────
    Llama 3.3 70B's built-in training knowledge
    Used when both Qdrant and web search fail
    Always gives some answer
    Clearly labeled as general knowledge
    """

    def __init__(self):
        logger.info("Initializing 3-tier RAG Pipeline...")

        self.retriever = LegalRetriever()
        self.llm = GroqLLM()
        self.web_searcher = WebSearcher()

        web_status = (
            "enabled" if self.web_searcher.enabled
            else "disabled (no API key)"
        )
        logger.info(f"Web search: {web_status}")
        logger.info("RAG Pipeline ready")

    def run(
        self,
        question: str,
        jurisdiction: Optional[str] = None,
        topic: Optional[str] = None,
        top_k: int = None,
    ) -> Dict:

        top_k = top_k or settings.TOP_K_RESULTS
        question = question.strip()

        # validate
        if not question:
            return self._error_response(
                "Question cannot be empty",
                jurisdiction, topic
            )

        if len(question) > settings.MAX_QUESTION_LENGTH:
            return self._error_response(
                f"Question too long. Max "
                f"{settings.MAX_QUESTION_LENGTH} chars.",
                jurisdiction, topic
            )

        logger.info(
            f"Pipeline: '{question[:60]}' "
            f"| {jurisdiction} | {topic}"
        )

        # ══════════════════════════════════════════════
        # TIER 1: Try our Qdrant database
        # ══════════════════════════════════════════════
        sources = self._try_qdrant(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic,
            top_k=top_k,
        )

        if sources:
            logger.info(
                f"TIER 1 SUCCESS: {len(sources)} docs from Qdrant"
            )
            return self._generate_from_qdrant(
                question=question,
                sources=sources,
                jurisdiction=jurisdiction,
                topic=topic,
            )

        # ══════════════════════════════════════════════
        # TIER 2: Try web search
        # ══════════════════════════════════════════════
        logger.info(
            "TIER 1 MISS: No Qdrant results. "
            "Trying web search..."
        )

        web_results = self._try_web_search(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic,
        )

        if web_results:
            logger.info(
                f"TIER 2 SUCCESS: {len(web_results)} web results"
            )
            return self._generate_from_web(
                question=question,
                web_results=web_results,
                jurisdiction=jurisdiction,
                topic=topic,
            )

        # ══════════════════════════════════════════════
        # TIER 3: LLM general knowledge
        # ══════════════════════════════════════════════
        logger.info(
            "TIER 2 MISS: No web results. "
            "Using LLM general knowledge..."
        )

        return self._generate_from_llm_knowledge(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic,
        )

    # ──────────────────────────────────────────────────
    # TIER 1 METHODS
    # ──────────────────────────────────────────────────
    def _try_qdrant(
        self,
        question: str,
        jurisdiction: Optional[str],
        topic: Optional[str],
        top_k: int,
    ) -> list:
        """
        Try Qdrant with progressive filter relaxation.
        First try: jurisdiction + topic
        Second try: jurisdiction only
        Third try: no filters (all documents)
        """

        # attempt 1: full filters
        if jurisdiction or topic:
            sources = self.retriever.retrieve(
                question=question,
                jurisdiction=jurisdiction,
                topic=topic,
                top_k=top_k,
            )
            if sources:
                return sources

        # attempt 2: jurisdiction only (drop topic filter)
        if jurisdiction and topic:
            logger.info(
                "Relaxing filter: dropping topic filter"
            )
            sources = self.retriever.retrieve(
                question=question,
                jurisdiction=jurisdiction,
                topic=None,
                top_k=top_k,
            )
            if sources:
                return sources

        # attempt 3: no filters at all
        if jurisdiction or topic:
            logger.info(
                "Relaxing filter: searching all documents"
            )
            sources = self.retriever.retrieve(
                question=question,
                jurisdiction=None,
                topic=None,
                top_k=top_k,
            )
            if sources:
                return sources

        return []

    def _generate_from_qdrant(
        self,
        question: str,
        sources: list,
        jurisdiction: Optional[str],
        topic: Optional[str],
    ) -> Dict:
        """Generate answer using Qdrant retrieved documents"""

        top_score = sources[0]["score"]
        is_low_confidence = top_score < 0.5

        context = build_context_string(sources)
        prompt = build_rag_prompt(
            question=question,
            context=context,
            is_low_confidence=is_low_confidence
        )

        try:
            answer = self.llm.generate(
                prompt=prompt,
                temperature=0.15
            )
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._error_response(
                f"AI error: {e}", jurisdiction, topic
            )

        return {
            "answer": answer,
            "sources": sources,
            "web_sources": [],
            "jurisdiction": jurisdiction,
            "topic": topic,
            "has_results": True,
            "answer_source": "indexed_documents",
            "top_score": top_score,
            "is_low_confidence": is_low_confidence,
            "disclaimer": DISCLAIMER,
        }

    # ──────────────────────────────────────────────────
    # TIER 2 METHODS
    # ──────────────────────────────────────────────────
    def _try_web_search(
        self,
        question: str,
        jurisdiction: Optional[str],
        topic: Optional[str],
    ) -> list:
        """Try web search for the question"""

        if not self.web_searcher.enabled:
            logger.info("Web search disabled, skipping tier 2")
            return []

        # focused search first
        results = self.web_searcher.search(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic,
            max_results=5,
        )

        if results:
            return results

        # general search fallback
        results = self.web_searcher.search_general(
            question=question,
            max_results=3,
        )

        return results

    def _generate_from_web(
        self,
        question: str,
        web_results: list,
        jurisdiction: Optional[str],
        topic: Optional[str],
    ) -> Dict:
        """Generate answer using web search results"""

        web_context = self.web_searcher.format_for_prompt(
            web_results
        )

        prompt = build_web_search_prompt(
            question=question,
            web_context=web_context,
        )

        try:
            answer = self.llm.generate(
                prompt=prompt,
                temperature=0.2
            )
        except Exception as e:
            logger.error(f"LLM error on web results: {e}")
            answer = (
                f"Found web results but could not generate "
                f"answer: {e}"
            )

        # format web results as sources for display
        web_sources_formatted = [
            {
                "text": r.get("content", "")[:300],
                "law_name": r.get("title", "Web Result"),
                "source_url": r.get("url", ""),
                "jurisdiction": jurisdiction or "General",
                "topic": topic or "general",
                "agency": "Web Search",
                "score": r.get("score", 0.5),
                "effective_date": "",
                "document_type": "web",
            }
            for r in web_results[:3]
        ]

        return {
            "answer": answer,
            "sources": web_sources_formatted,
            "web_sources": web_results,
            "jurisdiction": jurisdiction,
            "topic": topic,
            "has_results": True,
            "answer_source": "web_search",
            "is_low_confidence": False,
            "disclaimer": DISCLAIMER,
        }

    # ──────────────────────────────────────────────────
    # TIER 3 METHODS
    # ──────────────────────────────────────────────────
    def _generate_from_llm_knowledge(
        self,
        question: str,
        jurisdiction: Optional[str],
        topic: Optional[str],
    ) -> Dict:
        """
        Use LLM general training knowledge.
        Always gives some answer.
        Clearly labeled as general knowledge.
        """

        prompt = build_llm_knowledge_prompt(
            question=question,
            jurisdiction=jurisdiction or "Central",
            topic=topic or "employment law",
        )

        try:
            answer = self.llm.generate(
                prompt=prompt,
                temperature=0.3,
            )
        except Exception as e:
            logger.error(f"LLM knowledge error: {e}")
            answer = (
                "I encountered an error generating an answer. "
                "Please try rephrasing your question or "
                "check the official government websites."
            )

        return {
            "answer": answer,
            "sources": [],
            "web_sources": [],
            "jurisdiction": jurisdiction,
            "topic": topic,
            "has_results": False,
            "answer_source": "llm_knowledge",
            "is_low_confidence": True,
            "disclaimer": DISCLAIMER,
        }

    # ──────────────────────────────────────────────────
    # COMPARISON
    # ──────────────────────────────────────────────────
    def run_comparison(
        self,
        question: str,
        jurisdiction1: str,
        jurisdiction2: str,
        topic: Optional[str] = None,
    ) -> Dict:

        logger.info(
            f"Comparison: {jurisdiction1} vs {jurisdiction2}"
        )

        result1 = self.run(
            question=question,
            jurisdiction=jurisdiction1,
            topic=topic,
            top_k=3
        )
        result2 = self.run(
            question=question,
            jurisdiction=jurisdiction2,
            topic=topic,
            top_k=3
        )

        return {
            "question": question,
            "topic": topic,
            "comparison": {
                jurisdiction1: {
                    "answer": result1["answer"],
                    "sources": result1["sources"],
                    "has_results": result1["has_results"],
                    "answer_source": result1.get(
                        "answer_source", "unknown"
                    ),
                },
                jurisdiction2: {
                    "answer": result2["answer"],
                    "sources": result2["sources"],
                    "has_results": result2["has_results"],
                    "answer_source": result2.get(
                        "answer_source", "unknown"
                    ),
                }
            },
            "disclaimer": DISCLAIMER,
        }

    def _error_response(
        self, message, jurisdiction, topic
    ):
        return {
            "answer": message,
            "sources": [],
            "web_sources": [],
            "jurisdiction": jurisdiction,
            "topic": topic,
            "has_results": False,
            "answer_source": "error",
            "disclaimer": DISCLAIMER,
        }