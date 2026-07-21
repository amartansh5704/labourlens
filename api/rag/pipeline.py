# api/rag/pipeline.py
# Add reranker between retrieval and generation

from api.rag.retriever import LegalRetriever
from api.rag.llm import GroqLLM
from api.rag.reranker import LegalReranker
from api.rag.web_search import WebSearcher
from api.rag.prompts import (
    build_rag_prompt,
    build_context_string,
    build_no_results_message,
    build_web_search_prompt,
    build_llm_knowledge_prompt,
)
from api.core.config import settings
from shared.constants import DISCLAIMER
from loguru import logger
from typing import Optional, Dict


class LegalRAGPipeline:
    """
    3-tier RAG pipeline WITH reranking:

    TIER 1: Qdrant (fetch 20) → Reranker (pick 5) → LLM
    TIER 2: Web search → LLM
    TIER 3: LLM general knowledge
    """

    def __init__(self):
        logger.info("Initializing RAG Pipeline with Reranker...")

        self.retriever = LegalRetriever()
        self.llm = GroqLLM()
        self.web_searcher = WebSearcher()

        # initialize reranker
        if settings.ENABLE_RERANKER:
            self.reranker = LegalReranker()
            if self.reranker.enabled:
                logger.info("Reranker: ENABLED")
            else:
                logger.warning("Reranker: DISABLED (load failed)")
        else:
            self.reranker = None
            logger.info("Reranker: DISABLED (config)")

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

        if not question:
            return self._error_response(
                "Question cannot be empty",
                jurisdiction, topic
            )

        if len(question) > settings.MAX_QUESTION_LENGTH:
            return self._error_response(
                f"Question too long. "
                f"Max {settings.MAX_QUESTION_LENGTH} chars.",
                jurisdiction, topic
            )

        logger.info(
            f"Pipeline: '{question[:60]}' "
            f"| {jurisdiction} | {topic}"
        )

        # ── TIER 1: Qdrant + Reranker ─────────────────
        sources = self._try_qdrant_with_reranking(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic,
            top_k=top_k,
        )

        if sources:
            logger.info(
                f"TIER 1 SUCCESS: {len(sources)} docs"
            )
            return self._generate_from_qdrant(
                question=question,
                sources=sources,
                jurisdiction=jurisdiction,
                topic=topic,
            )

        # ── TIER 2: Web Search ─────────────────────────
        logger.info("TIER 1 MISS → trying web search...")
        web_results = self._try_web_search(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic,
        )

        if web_results:
            logger.info(
                f"TIER 2 SUCCESS: {len(web_results)} results"
            )
            return self._generate_from_web(
                question=question,
                web_results=web_results,
                jurisdiction=jurisdiction,
                topic=topic,
            )

        # ── TIER 3: LLM Knowledge ─────────────────────
        logger.info("TIER 2 MISS → using LLM knowledge...")
        return self._generate_from_llm_knowledge(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic,
        )

    # ──────────────────────────────────────────────────
    # TIER 1 - Qdrant with Reranking
    # ──────────────────────────────────────────────────
    def _try_qdrant_with_reranking(
        self,
        question: str,
        jurisdiction: Optional[str],
        topic: Optional[str],
        top_k: int,
    ) -> list:
        """
        Fetch candidates from Qdrant then rerank them.
        Tries progressively relaxed filters if nothing found.
        """

        # attempt 1: full filters
        candidates = self.retriever.retrieve(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic,
            top_k=top_k,
        )

        # attempt 2: drop topic filter
        if not candidates and topic and jurisdiction:
            logger.info("Relaxing: dropping topic filter")
            candidates = self.retriever.retrieve(
                question=question,
                jurisdiction=jurisdiction,
                topic=None,
                top_k=top_k,
            )

        # attempt 3: no filters
        if not candidates and (jurisdiction or topic):
            logger.info("Relaxing: no filters")
            candidates = self.retriever.retrieve(
                question=question,
                jurisdiction=None,
                topic=None,
                top_k=top_k,
            )

        if not candidates:
            return []

        # ── RERANKING STEP ─────────────────────────────
        if (
            self.reranker and
            self.reranker.enabled and
            len(candidates) > top_k
        ):
            logger.info(
                f"Reranking {len(candidates)} candidates "
                f"→ top {top_k}"
            )
            final_chunks = self.reranker.rerank(
                question=question,
                chunks=candidates,
                top_k=top_k,
            )
            logger.info(
                f"After reranking: {len(final_chunks)} chunks"
            )
        else:
            # no reranking, just take top_k
            final_chunks = candidates[:top_k]

        return final_chunks

    def _generate_from_qdrant(
        self,
        question: str,
        sources: list,
        jurisdiction: Optional[str],
        topic: Optional[str],
    ) -> Dict:
        """Generate answer from Qdrant + reranked chunks"""

        # use rerank_score if available, else qdrant score
        top_score = (
            sources[0].get("rerank_score") or
            sources[0].get("score", 0)
        )

        # low confidence if top score is low
        is_low_confidence = top_score < 0.3

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

        # add reranking info to sources for display
        for source in sources:
            if "rerank_score" in source:
                source["score"] = round(
                    source["rerank_score"], 4
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
    # TIER 2 - Web Search
    # ──────────────────────────────────────────────────
    def _try_web_search(
        self,
        question: str,
        jurisdiction: Optional[str],
        topic: Optional[str],
    ) -> list:
        if not self.web_searcher.enabled:
            return []

        results = self.web_searcher.search(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic,
            max_results=5,
        )

        if results:
            return results

        return self.web_searcher.search_general(
            question=question,
            max_results=3,
        )

    def _generate_from_web(
        self,
        question: str,
        web_results: list,
        jurisdiction: Optional[str],
        topic: Optional[str],
    ) -> Dict:
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
            logger.error(f"LLM error on web: {e}")
            answer = f"Web search found results but AI failed: {e}"

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
    # TIER 3 - LLM Knowledge
    # ──────────────────────────────────────────────────
    def _generate_from_llm_knowledge(
        self,
        question: str,
        jurisdiction: Optional[str],
        topic: Optional[str],
    ) -> Dict:
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
                "I encountered an error. "
                "Please try rephrasing your question."
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

    def _error_response(self, message, jurisdiction, topic):
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