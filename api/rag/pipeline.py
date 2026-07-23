# api/rag/pipeline.py
# Smart pipeline with conversation memory
# Short precise answers with expandable sections
# ALL tiers contribute to EVERY answer

from api.rag.retriever import LegalRetriever
from api.rag.llm import GroqLLM
from api.core.config import settings
from shared.constants import DISCLAIMER
from loguru import logger
from typing import Optional, Dict, List

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
    Combined pipeline with conversation memory.
    Uses ALL sources every time.
    Returns short precise structured answers.
    """

    def __init__(self):
        logger.info("Initializing RAG Pipeline...")

        self.retriever = LegalRetriever()
        self.llm = GroqLLM()

        # web search
        if WEB_SEARCH_AVAILABLE:
            try:
                self.web_searcher = WebSearcher()
                if self.web_searcher.enabled:
                    logger.info("Web search: ENABLED")
                else:
                    logger.info(
                        "Web search: DISABLED (no key)"
                    )
            except Exception as e:
                logger.warning(
                    f"Web search init failed: {e}"
                )
                self.web_searcher = None
        else:
            self.web_searcher = None
            logger.info("Web search: NOT AVAILABLE")

        # reranker
        self.reranker = None
        if RERANKER_AVAILABLE:
            try:
                enable = getattr(
                    settings, "ENABLE_RERANKER", False
                )
                if enable:
                    self.reranker = LegalReranker()
                    if not self.reranker.enabled:
                        self.reranker = None
                    else:
                        logger.info("Reranker: ENABLED")
            except Exception as e:
                logger.warning(
                    f"Reranker init failed: {e}"
                )
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
        chat_history: List[Dict] = None,
    ) -> Dict:

        top_k = top_k or settings.TOP_K_RESULTS
        question = question.strip()
        chat_history = chat_history or []

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
            f"| topic={topic} "
            f"| history={len(chat_history)} msgs"
        )

        # ── Step 1: Search Qdrant ──────────────────────
        qdrant_sources = self._get_qdrant_sources(
            question=question,
            jurisdiction=jurisdiction,
            topic=topic,
            top_k=top_k,
        )
        logger.info(
            f"Qdrant results: {len(qdrant_sources)}"
        )

        # ── Step 2: Web search ─────────────────────────
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
            chat_history=chat_history,
        )

        logger.info(
            f"Prompt: {len(prompt)} chars | "
            f"Qdrant={len(qdrant_sources)} "
            f"Web={len(web_sources)}"
        )

        # ── Step 4: Generate answer ────────────────────
        try:
            answer = self.llm.generate(
                prompt=prompt,
                temperature=0.1
            )
            logger.info("Answer generated")
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return self._error_response(
                f"AI error: {e}", jurisdiction, topic
            )

        # ── Step 5: Source label ───────────────────────
        if qdrant_sources and web_sources:
            answer_source = "all_sources"
        elif qdrant_sources:
            answer_source = "indexed_documents"
        elif web_sources:
            answer_source = "web_search"
        else:
            answer_source = "llm_knowledge"

        # ── Step 6: Combine display sources ───────────
        all_display_sources = list(qdrant_sources)
        for ws in web_sources[:2]:
            all_display_sources.append({
                "text": ws.get("content", "")[:300],
                "law_name": ws.get("title", "Web"),
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
            f"sources={len(all_display_sources)}"
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
    # TIER 1 - QDRANT
    # ══════════════════════════════════════════════════
    def _get_qdrant_sources(
        self,
        question: str,
        jurisdiction: Optional[str],
        topic: Optional[str],
        top_k: int,
    ) -> list:

        try:
            # attempt 1: full filters
            sources = self.retriever.retrieve(
                question=question,
                jurisdiction=jurisdiction,
                topic=topic,
                top_k=top_k,
            )

            # attempt 2: drop topic
            if not sources and topic and jurisdiction:
                logger.info("Qdrant: dropping topic filter")
                sources = self.retriever.retrieve(
                    question=question,
                    jurisdiction=jurisdiction,
                    topic=None,
                    top_k=top_k,
                )

            # attempt 3: no filters
            if not sources and (jurisdiction or topic):
                logger.info("Qdrant: no filters")
                sources = self.retriever.retrieve(
                    question=question,
                    jurisdiction=None,
                    topic=None,
                    top_k=top_k,
                )

            # rerank
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
            logger.error(f"Qdrant error: {e}")
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

            return results or []

        except Exception as e:
            logger.warning(f"Web search error: {e}")
            return []

    # ══════════════════════════════════════════════════
    # COMBINED PROMPT WITH MEMORY
    # ══════════════════════════════════════════════════
    def _build_combined_prompt(
        self,
        question: str,
        qdrant_sources: list,
        web_sources: list,
        jurisdiction: Optional[str],
        topic: Optional[str],
        chat_history: List[Dict] = None,
    ) -> str:

        sections = []

        # ── Conversation history ───────────────────────
        if chat_history:
            sections.append(
                "=== CONVERSATION HISTORY ==="
            )
            # last 6 messages only to save tokens
            recent = chat_history[-6:]
            for msg in recent:
                role = msg.get("role", "")
                content = str(
                    msg.get("content", "")
                )[:200]
                if role == "user":
                    sections.append(f"User: {content}")
                elif role == "assistant":
                    # only first line of assistant msg
                    first_line = content.split(
                        "\n"
                    )[0][:150]
                    sections.append(
                        f"Assistant: {first_line}"
                    )
            sections.append("")

        # ── Qdrant documents ───────────────────────────
        if qdrant_sources:
            sections.append(
                "=== INDEXED LEGAL DOCUMENTS ==="
            )
            for i, src in enumerate(
                qdrant_sources[:3], 1
            ):
                sections.append(
                    f"[Doc {i}] "
                    f"{src.get('law_name','Unknown')} "
                    f"({src.get('jurisdiction','')}):\n"
                    f"{src.get('text','')[:350]}"
                )
        else:
            sections.append(
                "=== DOCUMENTS ===\n"
                "No matching Indian law documents found."
            )

        # ── Web results ────────────────────────────────
        if web_sources:
            sections.append(
                "\n=== WEB SEARCH RESULTS ==="
            )
            for i, ws in enumerate(web_sources[:2], 1):
                sections.append(
                    f"[Web {i}] "
                    f"{ws.get('title','')}:\n"
                    f"{ws.get('content','')[:250]}"
                )
        else:
            sections.append(
                "\n=== WEB SEARCH ===\nNo results."
            )

        # ── Strict instructions ───────────────────────
        sections.append(f"""
=== CURRENT QUESTION ===
{question}

=== OUTPUT FORMAT ===
Respond in EXACTLY this structure:

ANSWER: [Answer here - length depends on question complexity]

KEY_DETAILS:
- [Specific fact with number/date]
- [Add more bullets if question needs detail]
- [No limit on bullets if user asks for details]

SOURCES_USED:
- [📄 Document name if used]
- [🌐 Web source if used]
- [🧠 AI knowledge if used]

LEGAL_BASIS: [Law name • Jurisdiction • Year]

=== LENGTH GUIDE ===
- "What is X?" → ANSWER is 1 sentence
- "How does X work?" → ANSWER is 2-3 sentences
- "Explain X" or "Tell me about X" → ANSWER is 1 paragraph
- "Give details" or "More info" or "Elaborate" → ANSWER is full detailed paragraph
- "Compare X and Y" → ANSWER covers both with comparison
- Follow-up questions → expand on previous answer context

=== STRICT RULES ===
1. Match answer length to question complexity
2. Only state numbers found in documents above
3. If number not in docs: write "verify at official source"
4. If user refers to previous messages: acknowledge and build on it
5. If about non-Indian law: use your knowledge, label [🧠]
6. Never invent section numbers or rates
7. Always complete all 4 sections
8. KEY_DETAILS can have as many bullets as needed for the question""")

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
        chat_history: List[Dict] = None,
    ) -> Dict:

        logger.info(
            f"Comparison: {jurisdiction1} vs "
            f"{jurisdiction2}"
        )

        result1 = self.run(
            question=question,
            jurisdiction=jurisdiction1,
            topic=topic,
            top_k=3,
            chat_history=chat_history,
        )
        result2 = self.run(
            question=question,
            jurisdiction=jurisdiction2,
            topic=topic,
            top_k=3,
            chat_history=chat_history,
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