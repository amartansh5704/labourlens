# api/rag/pipeline.py
# Smart pipeline - always combines all available sources
# Tier 1: Qdrant indexed documents
# Tier 2: Web search (Tavily)
# Tier 3: LLM general knowledge
# ALL tiers contribute to EVERY answer

from api.rag.retriever import LegalRetriever
from api.rag.llm import GroqLLM
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
    logger.warning("WebSearcher not available")

try:
    from api.rag.reranker import LegalReranker
    RERANKER_AVAILABLE = True
except Exception:
    RERANKER_AVAILABLE = False
    logger.warning("Reranker not available")


class LegalRAGPipeline:
    """
    Combined pipeline that uses ALL available sources
    for every question to give the best possible answer.

    Flow for EVERY question:
    1. Search Qdrant for relevant indexed documents
    2. Search web via Tavily for current information
    3. Build ONE prompt with ALL gathered info
    4. LLM uses docs + web + own knowledge to answer
    """

    def __init__(self):
        logger.info("Initializing Combined RAG Pipeline...")

        self.retriever = LegalRetriever()
        self.llm = GroqLLM()

        # web search (optional)
        if WEB_SEARCH_AVAILABLE:
            try:
                self.web_searcher = WebSearcher()
                if self.web_searcher.enabled:
                    logger.info("Web search: ENABLED")
                else:
                    logger.info(
                        "Web search: DISABLED (no API key)"
                    )
            except Exception as e:
                logger.warning(f"Web search init failed: {e}")
                self.web_searcher = None
        else:
            self.web_searcher = None
            logger.info("Web search: NOT AVAILABLE")

        # reranker (optional)
        self.reranker = None
        if RERANKER_AVAILABLE:
            try:
                enable = getattr(
                    settings, "ENABLE_RERANKER", False
                )
                if enable:
                    self.reranker = LegalReranker()
                    if self.reranker.enabled:
                        logger.info("Reranker: ENABLED")
                    else:
                        self.reranker = None
            except Exception as e:
                logger.warning(f"Reranker init failed: {e}")
                self.reranker = None

        logger.info("RAG Pipeline ready")

    # ══════════════════════════════════════════════════
    # MAIN RUN METHOD
    # ══════════════════════════════════════════════════
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

        max_len = getattr(
            settings, "MAX_QUESTION_LENGTH", 500
        )
        if len(question) > max_len:
            return self._error_response(
                f"Question too long. Max {max_len} chars.",
                jurisdiction, topic
            )

        logger.info(
            f"Pipeline START: '{question[:60]}' "
            f"| jurisdiction={jurisdiction} "
            f"| topic={topic}"
        )

        # ── Step 1: Always search Qdrant ───────────────
        qdrant_sources = self._get_qdrant_sources(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic,
            top_k=top_k,
        )
        logger.info(
            f"Qdrant results: {len(qdrant_sources)}"
        )

        # ── Step 2: Always search web ──────────────────
        web_sources = self._get_web_sources(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic,
        )
        logger.info(
            f"Web results: {len(web_sources)}"
        )

        # ── Step 3: Build combined prompt ──────────────
        prompt = self._build_combined_prompt(
            question=question,
            qdrant_sources=qdrant_sources,
            web_sources=web_sources,
            jurisdiction=jurisdiction,
            topic=topic,
        )

        logger.info(
            f"Prompt built: {len(prompt)} chars | "
            f"Qdrant={len(qdrant_sources)} "
            f"Web={len(web_sources)}"
        )

        # ── Step 4: Generate answer ────────────────────
        try:
            answer = self.llm.generate(
                prompt=prompt,
                temperature=0.15
            )
            logger.info("Answer generated successfully")
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._error_response(
                f"AI error: {e}", jurisdiction, topic
            )

        # ── Step 5: Determine source label ─────────────
        if qdrant_sources and web_sources:
            answer_source = "all_sources"
        elif qdrant_sources:
            answer_source = "indexed_documents"
        elif web_sources:
            answer_source = "web_search"
        else:
            answer_source = "llm_knowledge"

        # ── Step 6: Combine sources for display ────────
        all_display_sources = list(qdrant_sources)

        for ws in web_sources[:2]:
            all_display_sources.append({
                "text": ws.get("content", "")[:300],
                "law_name": ws.get(
                    "title", "Web Result"
                ),
                "source_url": ws.get("url", ""),
                "jurisdiction": (
                    jurisdiction or "General"
                ),
                "topic": topic or "general",
                "agency": "🌐 Web Search",
                "score": float(ws.get("score", 0.5)),
                "effective_date": "",
                "document_type": "web",
                "chunk_index": 0,
                "file_type": "web",
                "title": ws.get("title", ""),
            })

        logger.info(
            f"Pipeline DONE: source={answer_source} | "
            f"display_sources={len(all_display_sources)}"
        )

        return {
            "answer": answer,
            "sources": all_display_sources,
            "web_sources": web_sources,
            "jurisdiction": jurisdiction,
            "topic": topic,
            "has_results": True,
            "answer_source": answer_source,
            "top_score": (
                qdrant_sources[0]["score"]
                if qdrant_sources else 0.0
            ),
            "is_low_confidence": (
                not qdrant_sources and not web_sources
            ),
            "disclaimer": DISCLAIMER,
        }

    # ══════════════════════════════════════════════════
    # TIER 1 - QDRANT SEARCH
    # ══════════════════════════════════════════════════
    def _get_qdrant_sources(
        self,
        question: str,
        jurisdiction: Optional[str],
        topic: Optional[str],
        top_k: int,
    ) -> list:
        """
        Search Qdrant with progressive filter relaxation.
        Always returns whatever is found.
        """

        try:
            # attempt 1: full filters
            sources = self.retriever.retrieve(
                question=question,
                jurisdiction=jurisdiction,
                topic=topic,
                top_k=top_k,
            )

            # attempt 2: drop topic filter
            if not sources and topic and jurisdiction:
                logger.info(
                    "Qdrant: relaxing topic filter"
                )
                sources = self.retriever.retrieve(
                    question=question,
                    jurisdiction=jurisdiction,
                    topic=None,
                    top_k=top_k,
                )

            # attempt 3: no filters
            if not sources and (jurisdiction or topic):
                logger.info(
                    "Qdrant: searching without filters"
                )
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

            return sources or []

        except Exception as e:
            logger.error(f"Qdrant search error: {e}")
            return []

    # ══════════════════════════════════════════════════
    # TIER 2 - WEB SEARCH
    # ══════════════════════════════════════════════════
    def _get_web_sources(
        self,
        question: str,
        jurisdiction: Optional[str],
        topic: Optional[str],
    ) -> list:
        """
        Search web via Tavily.
        Always tries even if Qdrant returned results.
        """

        if (
            not self.web_searcher or
            not self.web_searcher.enabled
        ):
            logger.debug(
                "Web search skipped: not enabled"
            )
            return []

        try:
            # focused search first
            results = self.web_searcher.search(
                question=question,
                jurisdiction=jurisdiction,
                topic=topic,
                max_results=3,
            )

            # general search if focused found nothing
            if not results:
                logger.info(
                    "Web: focused search empty, "
                    "trying general search"
                )
                results = self.web_searcher.search_general(
                    question=question,
                    max_results=3,
                )

            return results or []

        except Exception as e:
            logger.warning(f"Web search error: {e}")
            return []

    # ══════════════════════════════════════════════════
    # COMBINED PROMPT BUILDER
    # ══════════════════════════════════════════════════
    def _build_combined_prompt(
        self,
        question: str,
        qdrant_sources: list,
        web_sources: list,
        jurisdiction: Optional[str],
        topic: Optional[str],
    ) -> str:
        """
        Build precise, anti-hallucination prompt.
    Forces short answers and honest uncertainty.
        """

        sections = []

        # ── Qdrant documents section ───────────────────
        if qdrant_sources:
            sections.append(
                "=== INDEXED LEGAL DOCUMENTS ===\n"
                "Source: Verified Indian labor law database\n"
            )
            for i, src in enumerate(qdrant_sources, 1):
                sections.append(
                    f"[Document {i}]\n"
                    f"Law: {src.get('law_name', 'Unknown')}\n"
                    f"Jurisdiction: "
                    f"{src.get('jurisdiction', 'Unknown')}\n"
                    f"Agency: "
                    f"{src.get('agency', 'Unknown')}\n"
                    f"Content:\n{src.get('text', '')}\n"
                )
        else:
            sections.append(
                "=== INDEXED DOCUMENTS ===\n"
                "No matching documents found in our "
                "Indian legal database for this query.\n"
            )

        # ── Web search section ─────────────────────────
        if web_sources:
            sections.append(
                "\n=== LIVE WEB SEARCH RESULTS ===\n"
                "Source: Internet search (current info)\n"
            )
            for i, ws in enumerate(web_sources, 1):
                sections.append(
                    f"[Web Result {i}]\n"
                    f"Title: {ws.get('title', 'Unknown')}\n"
                    f"URL: {ws.get('url', '')}\n"
                    f"Content:\n"
                    f"{ws.get('content', '')[:500]}\n"
                )
        else:
            sections.append(
                "\n=== WEB SEARCH ===\n"
                "No web results available.\n"
            )

        # ── Main instruction section ───────────────────
        sections.append(f"""
=== YOUR INSTRUCTIONS ===

QUESTION: {question}

STRICT RULES FOR YOUR ANSWER:
1. Answer in 2-3 sentences MAXIMUM
2. Only state numbers/rates that appear in the documents above
3. If the exact number is not in documents, say "approximately X" or "check official source for current rate"
4. Do NOT list bullet points unless asked
5. Do NOT use headers
6. Do NOT say "based on the documents provided" - just answer
7. If about another country, answer briefly from your knowledge
8. End with the law name and jurisdiction in parentheses

ANSWER (2-3 sentences only):""")

        return "\n".join(sections)

    # ══════════════════════════════════════════════════
    # COMPARISON
    # ══════════════════════════════════════════════════
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

    # ══════════════════════════════════════════════════
    # ERROR RESPONSE
    # ══════════════════════════════════════════════════
    def _error_response(
        self,
        message: str,
        jurisdiction: Optional[str],
        topic: Optional[str],
    ) -> Dict:
        return {
            "answer": message,
            "sources": [],
            "web_sources": [],
            "jurisdiction": jurisdiction,
            "topic": topic,
            "has_results": False,
            "answer_source": "error",
            "is_low_confidence": True,
            "disclaimer": DISCLAIMER,
        }