# api/rag/prompts.py

# ─────────────────────────────────────────────────────────
# SYSTEM PROMPT
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are LaborLens, an expert employment law assistant with deep knowledge of Indian labor laws and global employment regulations.

YOUR BEHAVIOR:
- Match your answer length to what the user actually needs
- Short factual questions get short factual answers
- Requests for explanation, details, or elaboration get thorough comprehensive answers
- Never truncate an answer that needs to be complete
- Use your indexed documents as primary source
- Add web search results and your own knowledge to fill gaps
- Be conversational but professional

ANTI-HALLUCINATION RULES:
- Only state specific numbers/rates you find in the provided documents
- If a number is not in documents, say "approximately" or "verify at official source"
- Never invent law section numbers you are not sure about
- Clearly label sources: [📄 Document] [🌐 Web] [🧠 Knowledge]

NEVER:
- Say "I cannot answer this"
- Refuse employment law questions from any country
- Add filler like "Great question!" or "Certainly!"
- Truncate an answer that needs more detail"""


# ─────────────────────────────────────────────────────────
# SHORT ANSWER PROMPT - for simple factual questions
# ─────────────────────────────────────────────────────────
SHORT_ANSWER_PROMPT = """
=== CURRENT QUESTION ===
{question}

=== CONTEXT ===
{context}

=== TASK ===
Answer this simple factual question concisely.

ANSWER: [1-2 sentences with the direct fact]

KEY_DETAILS:
- [Key number or date if relevant]
- [One exception or note if important]

SOURCES_USED:
- [Source label]

LEGAL_BASIS: [Law • Jurisdiction • Year]"""


# ─────────────────────────────────────────────────────────
# DETAILED ANSWER PROMPT - for complex or detail requests
# ─────────────────────────────────────────────────────────
DETAILED_ANSWER_PROMPT = """
=== CURRENT QUESTION ===
{question}

=== CONTEXT ===
{context}

=== TASK ===
The user wants a DETAILED and COMPREHENSIVE answer.
Do NOT truncate. Cover all aspects thoroughly.

ANSWER:
[Write a complete detailed answer here.
Cover all relevant aspects.
Use multiple sentences and paragraphs if needed.
Include specific numbers, provisions, exceptions.
No word limit - be as thorough as the question requires.]

KEY_DETAILS:
- [Detail 1 with specific numbers/dates]
- [Detail 2]
- [Detail 3]
- [Detail 4]
- [Add as many as needed for completeness]

SOURCES_USED:
- [📄 Document name and what it says]
- [🌐 Web source if used]
- [🧠 AI knowledge areas used]

LEGAL_BASIS:
[Full law name • Jurisdiction • Year • Relevant sections if known]

IMPORTANT NOTE:
[Key exceptions, penalties, or practical implications]"""


# ─────────────────────────────────────────────────────────
# CONVERSATIONAL PROMPT - for follow-ups and chat questions
# ─────────────────────────────────────────────────────────
CONVERSATIONAL_PROMPT = """
=== CONVERSATION HISTORY ===
{history}

=== CURRENT QUESTION ===
{question}

=== CONTEXT ===
{context}

=== TASK ===
Answer this follow-up question in context of the conversation.
Build on what was previously discussed.
Length should match what the user is asking for.

ANSWER:
[Answer that acknowledges conversation context and builds on it]

KEY_DETAILS:
- [Relevant details]

SOURCES_USED:
- [Sources]

LEGAL_BASIS: [Law • Jurisdiction • Year]"""


# ─────────────────────────────────────────────────────────
# JURISDICTION URLS
# ─────────────────────────────────────────────────────────
JURISDICTION_URLS = {
    "Central": "https://labour.gov.in",
    "Delhi": "https://labour.delhi.gov.in",
    "Maharashtra": "https://mahakamgar.maharashtra.gov.in",
    "Karnataka": "https://labour.kar.nic.in",
    "Tamil Nadu": "https://labour.tn.gov.in",
    "Telangana": "https://labour.telangana.gov.in",
}


# ─────────────────────────────────────────────────────────
# QUESTION CLASSIFIER
# ─────────────────────────────────────────────────────────
def classify_question(
    question: str,
    chat_history: list = None,
) -> str:
    """
    Classify question to determine answer style needed.

    Returns:
        "short"          - simple factual question
        "detailed"       - user wants comprehensive answer
        "conversational" - follow-up or chat question
    """
    q = question.lower().strip()
    chat_history = chat_history or []

    # detect detail requests
    detail_triggers = [
        "explain", "elaborate", "detail", "details",
        "tell me more", "more info", "more information",
        "comprehensive", "thoroughly", "in depth",
        "in detail", "describe", "walk me through",
        "break down", "breakdown", "full explanation",
        "everything about", "all about", "what all",
        "deep dive", "deep-dive", "extensively",
        "give me a complete", "complete overview",
        "what are all", "list all", "list everything",
        "how does it work", "how does this work",
        "how do i", "what is the process",
        "step by step", "step-by-step",
        "what happens if", "what are the consequences",
        "penalties", "what are the rules",
        "rights", "entitlements", "benefits",
    ]

    for trigger in detail_triggers:
        if trigger in q:
            return "detailed"

    # detect conversational follow-ups
    follow_up_triggers = [
        "what about", "and what about",
        "you mentioned", "you said",
        "from above", "from before",
        "previous", "earlier",
        "that", "this", "it",
        "same for", "what did",
        "can you", "could you",
        "also", "additionally",
        "furthermore", "moreover",
        "follow up", "follow-up",
        "based on that", "related to that",
    ]

    has_history = len(chat_history) > 0
    is_follow_up = any(
        trigger in q for trigger in follow_up_triggers
    )

    if has_history and is_follow_up:
        return "conversational"

    # everything else is short
    return "short"


# ─────────────────────────────────────────────────────────
# CONTEXT BUILDER
# ─────────────────────────────────────────────────────────
def build_context_string(
    qdrant_sources: list,
    web_sources: list,
) -> str:
    """Build context from all sources"""
    parts = []

    if qdrant_sources:
        parts.append("INDEXED LEGAL DOCUMENTS:")
        for i, src in enumerate(qdrant_sources[:5], 1):
            parts.append(
                f"[Doc {i}] "
                f"{src.get('law_name', 'Unknown')} "
                f"({src.get('jurisdiction', '')}):\n"
                f"{src.get('text', '')[:500]}"
            )
    else:
        parts.append(
            "DOCUMENTS: No matching Indian law docs found."
        )

    if web_sources:
        parts.append("\nWEB SEARCH RESULTS:")
        for i, ws in enumerate(web_sources[:3], 1):
            parts.append(
                f"[Web {i}] {ws.get('title', '')}:\n"
                f"{ws.get('content', '')[:400]}"
            )

    return "\n\n".join(parts)


# ─────────────────────────────────────────────────────────
# MAIN PROMPT BUILDER
# ─────────────────────────────────────────────────────────
def build_prompt(
    question: str,
    qdrant_sources: list,
    web_sources: list,
    chat_history: list = None,
) -> str:
    """
    Build the right prompt based on question type.
    """
    chat_history = chat_history or []

    question_type = classify_question(
        question, chat_history
    )

    context = build_context_string(
        qdrant_sources, web_sources
    )

    if question_type == "detailed":
        return DETAILED_ANSWER_PROMPT.format(
            question=question,
            context=context,
        )

    elif question_type == "conversational":
        # format history
        history_parts = []
        for msg in chat_history[-6:]:
            role = msg.get("role", "")
            content = str(
                msg.get("content", "")
            )[:300]
            if role == "user":
                history_parts.append(
                    f"User: {content}"
                )
            elif role == "assistant":
                first_line = content.split(
                    "\n"
                )[0][:200]
                history_parts.append(
                    f"Assistant: {first_line}"
                )

        history_text = "\n".join(history_parts)

        return CONVERSATIONAL_PROMPT.format(
            question=question,
            context=context,
            history=history_text,
        )

    else:
        # short answer
        return SHORT_ANSWER_PROMPT.format(
            question=question,
            context=context,
        )


# ─────────────────────────────────────────────────────────
# KEEP OLD FUNCTIONS FOR COMPATIBILITY
# ─────────────────────────────────────────────────────────
def build_rag_prompt(
    question: str,
    context: str,
    is_low_confidence: bool = False,
) -> str:
    return SHORT_ANSWER_PROMPT.format(
        question=question,
        context=context,
    )


def build_context_str(sources: list) -> str:
    parts = []
    for i, src in enumerate(sources, 1):
        parts.append(
            f"[{i}] {src.get('law_name', '')} "
            f"({src.get('jurisdiction', '')}):\n"
            f"{src.get('text', '')[:400]}"
        )
    return "\n\n".join(parts)


def build_no_results_message(
    jurisdiction: str,
    question: str,
) -> str:
    url = JURISDICTION_URLS.get(
        jurisdiction, "https://labour.gov.in"
    )
    return (
        f"No specific documents found for "
        f"{jurisdiction}. "
        f"Try without filters or visit {url}"
    )


def build_llm_knowledge_prompt(
    question: str,
    jurisdiction: str = "Central",
    topic: str = "employment law",
) -> str:
    return SHORT_ANSWER_PROMPT.format(
        question=question,
        context="No documents found. Use your knowledge.",
    )


def build_web_search_prompt(
    question: str,
    web_context: str,
) -> str:
    return SHORT_ANSWER_PROMPT.format(
        question=question,
        context=f"WEB RESULTS:\n{web_context}",
    )


def build_no_docs_prompt(
    question: str,
    jurisdiction: str = "Central",
    topic: str = "employment law",
) -> str:
    return SHORT_ANSWER_PROMPT.format(
        question=question,
        context="No documents. Use general knowledge.",
    )