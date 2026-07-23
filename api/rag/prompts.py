# api/rag/prompts.py
# Smart prompts that combine documents + LLM knowledge

# ─────────────────────────────────────────────────────────
# SYSTEM PROMPT - Makes model intelligent and natural
# ─────────────────────────────────────────────────────────
SYSTEM_PROMPT = """You are LaborLens, a precise and helpful employment law assistant.

ANSWER LENGTH RULES:
- Simple factual question (what is X rate, what is X amount): 1-2 sentences
- Procedural question (how to do X, what is the process): 1 short paragraph
- Complex question (explain X, compare X and Y): 2-3 paragraphs
- User asks for details/more info/explain further: give full detailed answer
- User asks follow up on previous answer: expand on that specific point

ANTI-HALLUCINATION:
- Only state numbers that appear in documents
- If unsure about specific number: say "approximately" or "verify at official source"
- Never invent law section numbers
- Label clearly: [📄 Document] [🌐 Web] [🧠 Knowledge]

TONE: Direct, clear, professional. No filler phrases like "Great question" or "Certainly"."""


# ─────────────────────────────────────────────────────────
# LOW CONFIDENCE PROMPT
# ─────────────────────────────────────────────────────────
LOW_CONFIDENCE_RAG_PROMPT = """You are LaborLens, an expert employment law assistant.

The retrieved documents may be only partially relevant to this question. Use them where applicable, and supplement with your own knowledge.

RETRIEVED DOCUMENTS (may be partially relevant):
{context}

USER QUESTION: {question}

INSTRUCTIONS:
- Answer the question as best you can
- Use the documents where relevant
- Add your own knowledge to fill gaps
- Be transparent about what comes from documents vs your knowledge
- If the question is about a topic or country not in the documents, answer from your knowledge

**Answer:**
[Your best comprehensive answer]

**Sources Used:**
[What came from documents vs general knowledge]

---
*Always verify specific rates and dates with official sources.*"""


# ─────────────────────────────────────────────────────────
# WEB SEARCH PROMPT
# ─────────────────────────────────────────────────────────
WEB_SEARCH_PROMPT = """You are LaborLens, an expert employment law assistant.

Our database did not have specific documents, so we searched the internet:

WEB SEARCH RESULTS:
{web_context}

USER QUESTION: {question}

Answer using the web results above plus your own knowledge. Be specific with numbers and dates.

**Answer:**
[Clear answer from web results + knowledge]

**Key Details:**
[Specific numbers, dates, rates found]

**Sources:**
[List websites where info was found]

**⚠️ Verify:** This answer includes web search results. Always verify current rates at official sources.

---
*Not legal advice. Consult a qualified lawyer.*"""


# ─────────────────────────────────────────────────────────
# LLM KNOWLEDGE PROMPT
# ─────────────────────────────────────────────────────────
LLM_KNOWLEDGE_PROMPT = """You are LaborLens, an expert employment law assistant.

No specific documents or web results were found. Answer using your training knowledge.

USER QUESTION: {question}
JURISDICTION: {jurisdiction}
TOPIC: {topic}

Answer helpfully using your general knowledge. Be transparent about confidence level.

**Answer:**
[Your best answer from general knowledge]

**Confidence Level:** [High / Medium / Low]

**Verify At:**
[Recommend official website for verification]

**⚠️ Important:** This answer is from AI general knowledge. Verify with official sources.

---
*Not legal advice. Consult a qualified lawyer.*"""


# ─────────────────────────────────────────────────────────
# NO RESULTS MESSAGE
# ─────────────────────────────────────────────────────────
NO_RESULTS_MESSAGE = """I couldn't find specific indexed documents for this query in {jurisdiction}.

**What you can do:**
• Try without jurisdiction filter
• Rephrase with different keywords
• Visit [{official_url}]({official_url})

*Our database is continuously updated.*"""


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
# BUILDER FUNCTIONS
# ─────────────────────────────────────────────────────────
def build_rag_prompt(
    question: str,
    context: str,
    is_low_confidence: bool = False
) -> str:
    if is_low_confidence:
        return LOW_CONFIDENCE_RAG_PROMPT.format(
            context=context,
            question=question
        )
    return LEGAL_RAG_PROMPT.format(
        context=context,
        question=question
    )


def build_context_string(sources: list) -> str:
    """Convert retrieved chunks to formatted context"""
    if not sources:
        return "No documents retrieved."

    parts = []
    for i, source in enumerate(sources, 1):
        part = f"""[Document {i}]
Law: {source.get('law_name', 'Unknown')}
Jurisdiction: {source.get('jurisdiction', 'Unknown')}
Agency: {source.get('agency', 'Unknown')}
Effective: {source.get('effective_date', 'Not specified')}
Relevance: {source.get('score', 0):.0%}

{source.get('text', '')}"""
        parts.append(part)

    return "\n\n---\n\n".join(parts)


def build_web_search_prompt(
    question: str,
    web_context: str,
) -> str:
    return WEB_SEARCH_PROMPT.format(
        web_context=web_context,
        question=question,
    )


def build_no_docs_prompt(
    question: str,
    jurisdiction: str = "Central",
    topic: str = "employment law",
) -> str:
    return LLM_KNOWLEDGE_PROMPT.format(
        question=question,
        jurisdiction=jurisdiction or "Central India",
        topic=topic or "employment law",
    )


def build_llm_knowledge_prompt(
    question: str,
    jurisdiction: str = "Central",
    topic: str = "employment law",
) -> str:
    return LLM_KNOWLEDGE_PROMPT.format(
        question=question,
        jurisdiction=jurisdiction or "Central India",
        topic=topic or "employment law",
    )


def build_no_results_message(
    jurisdiction: str,
    question: str
) -> str:
    url = JURISDICTION_URLS.get(
        jurisdiction, "https://labour.gov.in"
    )
    return NO_RESULTS_MESSAGE.format(
        jurisdiction=jurisdiction or "the selected jurisdiction",
        official_url=url
    )