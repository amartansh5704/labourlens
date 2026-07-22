# api/rag/pipeline.py
# Smart pipeline - always combines all available sources
# Tier 1: Qdrant indexed documents
# Tier 2: Web search (Tavily)
# Tier 3: LLM general knowledge
# ALL tiers contribute to every answer

from api.rag.retriever import LegalRetriever
from api.rag.llm import GroqLLM
from api.rag.prompts import (
    build_context_string,
    build_no_results_message,
    SYSTEM_PROMPT,
)
from api.core.config import settings
from shared.constants import DISCLAIMER
from loguru import logger
from typing import Optional, Dict

# try importing optional modules
try:
    from api.rag.web_search import WebSearcher
    WEB_SEARCH_AVAILABLE = True
except Exception:
    WEB_SEARCH_AVAILABLE = False

try:
    from api.rag.reranker import LegalReranker
    RERANKER_AVAILABLE = True
except Exception:
    RERANKER_AVAILABLE = False


class LegalRAGPipeline:
    """
    Combined pipeline that uses ALL available sources
    for every question to give the best possible answer.
    """

    def __init__(self):
        logger.info("Initializing Combined RAG Pipeline...")

        self.retriever = LegalRetriever()
        self.llm = GroqLLM()

        # web search (optional)
        if WEB_SEARCH_AVAILABLE:
            self.web_searcher = WebSearcher()
            if self.web_searcher.enabled:
                logger.info("Web search: ENABLED")
            else:
                logger.info("Web search: DISABLED (no key)")
        else:
            self.web_searcher = None
            logger.info("Web search: NOT AVAILABLE")

        # reranker (optional)
        if RERANKER_AVAILABLE and settings.ENABLE_RERANKER:
            self.reranker = LegalReranker()
            if self.reranker.enabled:
                logger.info("Reranker: ENABLED")
            else:
                self.reranker = None
        else:
            self.reranker = None

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
                f"Question too long.",
                jurisdiction, topic
            )

        logger.info(
            f"Pipeline: '{question[:60]}' "
            f"| {jurisdiction} | {topic}"
        )

        # ══════════════════════════════════════════════
        # GATHER ALL SOURCES
        # ══════════════════════════════════════════════

        # TIER 1: Qdrant indexed documents
        qdrant_sources = self._get_qdrant_sources(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic,
            top_k=top_k,
        )

        # TIER 2: Web search
        web_sources = self._get_web_sources(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic,
        )

        # ══════════════════════════════════════════════
        # BUILD COMBINED PROMPT
        # ══════════════════════════════════════════════
        prompt = self._build_combined_prompt(
            question=question,
            qdrant_sources=qdrant_sources,
            web_sources=web_sources,
            jurisdiction=jurisdiction,
            topic=topic,
        )

        # ══════════════════════════════════════════════
        # GENERATE ANSWER (Tier 3 = LLM knowledge)
        # ══════════════════════════════════════════════
        try:
            answer = self.llm.generate(
                prompt=prompt,
                temperature=0.2
            )
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._error_response(
                f"AI error: {e}", jurisdiction, topic
            )

        # determine answer source
        if qdrant_sources and web_sources:
            answer_source = "all_sources"
        elif qdrant_sources:
            answer_source = "indexed_documents"
        elif web_sources:
            answer_source = "web_search"
        else:
            answer_source = "llm_knowledge"

        # combine all sources for display
        all_display_sources = qdrant_sources.copy()

        # add web sources formatted for display
        for ws in web_sources[:3]:
            all_display_sources.append({
                "text": ws.get("content", "")[:300],
                "law_name": ws.get(
                    "title", "Web Result"
                ),
                "source_url": ws.get("url", ""),
                "jurisdiction": jurisdiction or "General",
                "topic": topic or "general",
                "agency": "🌐 Web Search",
                "score": ws.get("score", 0.5),
                "effective_date": "",
                "document_type": "web",
            })

        top_score = (
            qdrant_sources[0]["score"]
            if qdrant_sources else 0
        )

        return {
            "answer": answer,
            "sources": all_display_sources,
            "web_sources": web_sources,
            "jurisdiction": jurisdiction,
            "topic": topic,
            "has_results": True,
            "answer_source": answer_source,
            "top_score": top_score,
            "is_low_confidence": (
                not qdrant_sources and not web_sources
            ),
            "disclaimer": DISCLAIMER,
        }

    # ──────────────────────────────────────────────────
    # TIER 1 - Qdrant Search
    # ──────────────────────────────────────────────────
    def _get_qdrant_sources(
        self,
        question: str,
        jurisdiction: Optional[str],
        topic: Optional[str],
        top_k: int,
    ) -> list:
        """Get relevant chunks from Qdrant"""

        # try with full filters first
        sources = self.retriever.retrieve(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic,
            top_k=top_k,
        )

        # relax filters if nothing found
        if not sources and topic and jurisdiction:
            sources = self.retriever.retrieve(
                question=question,
                jurisdiction=jurisdiction,
                topic=None,
                top_k=top_k,
            )

        if not sources and (jurisdiction or topic):
            sources = self.retriever.retrieve(
                question=question,
                jurisdiction=None,
                topic=None,
                top_k=top_k,
            )

        # rerank if available
        if (
            sources and
            self.reranker and
            self.reranker.enabled and
            len(sources) > top_k
        ):
            sources = self.reranker.rerank(
                question=question,
                chunks=sources,
                top_k=top_k,
            )

        if sources:
            logger.info(
                f"Qdrant: {len(sources)} results"
            )
        else:
            logger.info("Qdrant: no results")

        return sources

    # ──────────────────────────────────────────────────
    # TIER 2 - Web Search
    # ──────────────────────────────────────────────────
    def _get_web_sources(
        self,
        question: str,
        jurisdiction: Optional[str],
        topic: Optional[str],
    ) -> list:
        """Get web search results (always try)"""

        if (
            not self.web_searcher or
            not self.web_searcher.enabled
        ):
            return []

        try:
            results = self.web_searcher.search(
                question=question,
                jurisdiction=jurisdiction,
                topic=topic,
                max_results=3,
            )

            if not results:
                results = self.web_searcher.search_general(
                    question=question,
                    max_results=3,
                )

            if results:
                logger.info(
                    f"Web search: {len(results)} results"
                )
            else:
                logger.info("Web search: no results")

            return results

        except Exception as e:
            logger.warning(f"Web search failed: {e}")
            return []

    # ──────────────────────────────────────────────────
    # COMBINED PROMPT BUILDER
    # ──────────────────────────────────────────────────
    def _build_combined_prompt(
        self,
        question: str,
        qdrant_sources: list,
        web_sources: list,
        jurisdiction: Optional[str],
        topic: Optional[str],
    ) -> str:
        """
        Build one prompt that includes:
        - Retrieved documents (if any)
        - Web search results (if any)
        - Instructions to use LLM knowledge too
        """

        parts = []

        parts.append(
            "You are LaborLens. Answer the following "
            "question using ALL available information.\n"
        )

        # add Qdrant documents
        if qdrant_sources:
            parts.append(
                "═══ INDEXED LEGAL DOCUMENTS ═══\n"
                "(From our verified legal document database)\n"
            )
            for i, src in enumerate(qdrant_sources, 1):
                parts.append(
                    f"[Document {i}]\n"
                    f"Law: {src.get('law_name', 'Unknown')}\n"
                    f"Jurisdiction: "
                    f"{src.get('jurisdiction', 'Unknown')}\n"
                    f"Agency: "
                    f"{src.get('agency', 'Unknown')}\n"
                    f"Relevance: "
                    f"{src.get('score', 0):.0%}\n\n"
                    f"{src.get('text', '')}\n"
                )
        else:
            parts.append(
                "═══ NO INDEXED DOCUMENTS FOUND ═══\n"
                "Our database did not have matching "
                "documents for this question.\n"
            )

        # add web search results
        if web_sources:
            parts.append(
                "\n═══ WEB SEARCH RESULTS ═══\n"
                "(From live internet search)\n"
            )
            for i, ws in enumerate(web_sources, 1):
                parts.append(
                    f"[Web Result {i}]\n"
                    f"Title: "
                    f"{ws.get('title', 'Unknown')}\n"
                    f"URL: {ws.get('url', '')}\n"
                    f"Content: "
                    f"{ws.get('content', '')}\n"
                )

        # add instructions
        parts.append(f"""
═══ YOUR TASK ═══

USER QUESTION: {question}

INSTRUCTIONS:
1. Use the indexed legal documents as PRIMARY source (most trusted)
2. Use web search results as SECONDARY source (current info)
3. Use your own training knowledge as TERTIARY source (fill gaps)
4. If the question is about a topic NOT covered by documents (like another country), use your knowledge and say so clearly
5. ALWAYS give a helpful, complete answer - never say "I cannot answer"
6. Show which information came from which source
7. Be specific with numbers, dates, and provisions
8. If the question mentions a country or topic outside Indian law, answer it using your knowledge

FORMAT:

**Answer:**
[Complete answer combining all sources]

**Key Details:**
[Bullet points with specifics]

**Sources Used:**
[Which info came from: 📄 Documents / 🌐 Web Search / 🧠 AI Knowledge]

**Legal Basis:**
[Relevant laws and jurisdictions]

**Important Note:**
[Caveats, verify recommendations]

---
*Not legal advice. Consult a qualified lawyer.*
""")

        return "\n".join(parts)

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
            f"Comparison: {jurisdiction1} vs "
            f"{jurisdiction2}"
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