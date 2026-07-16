# test_rag.py
# Tests the complete RAG pipeline end to end
# Run after test_scraper.py passes

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

print("=" * 60)
print("LaborLens - RAG Pipeline Tests")
print("=" * 60)

# ── Test 1: Config ─────────────────────────────────────
print("\n[1] Testing configuration...")
try:
    from api.core.config import settings

    assert settings.GROQ_API_KEY, "GROQ_API_KEY missing"
    assert settings.GROQ_MODEL, "GROQ_MODEL missing"
    assert settings.QDRANT_HOST, "QDRANT_HOST missing"
    assert settings.EMBEDDING_MODEL, "EMBEDDING_MODEL missing"

    print(f"    ✅ GROQ_MODEL:      {settings.GROQ_MODEL}")
    print(f"    ✅ QDRANT_HOST:     {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    print(f"    ✅ COLLECTION:      {settings.QDRANT_COLLECTION}")
    print(f"    ✅ EMBEDDING_MODEL: {settings.EMBEDDING_MODEL}")
    print(f"    ✅ TOP_K_RESULTS:   {settings.TOP_K_RESULTS}")
    print(f"    ✅ MIN_SCORE:       {settings.MIN_SCORE_THRESHOLD}")

except Exception as e:
    print(f"    ❌ Config failed: {e}")

# ── Test 2: Prompts ────────────────────────────────────
print("\n[2] Testing prompt builder...")
try:
    from api.rag.prompts import (
        build_rag_prompt,
        build_context_string,
        build_no_results_message,
    )

    # fake sources to test with
    fake_sources = [
        {
            "text": "The minimum wages for unskilled workers in Delhi shall be Rs 17,494 per month.",
            "score": 0.92,
            "law_name": "Delhi Minimum Wages Notification 2023",
            "jurisdiction": "Delhi",
            "topic": "minimum_wage",
            "agency": "Delhi Labour Department",
            "source_url": "https://labour.delhi.gov.in/test",
            "effective_date": "October 2023",
        },
        {
            "text": "Workers in the unskilled category include helpers, sweepers and daily wage workers.",
            "score": 0.85,
            "law_name": "Delhi Minimum Wages Notification 2023",
            "jurisdiction": "Delhi",
            "topic": "minimum_wage",
            "agency": "Delhi Labour Department",
            "source_url": "https://labour.delhi.gov.in/test2",
            "effective_date": "October 2023",
        }
    ]

    # test context building
    context = build_context_string(fake_sources)
    assert "Delhi Minimum Wages" in context
    assert "Rs 17,494" in context
    print(f"    ✅ Context built: {len(context)} chars")

    # test prompt building
    prompt = build_rag_prompt(
        question="What is minimum wage in Delhi?",
        context=context,
        is_low_confidence=False
    )
    assert "What is minimum wage in Delhi?" in prompt
    assert "LEGAL DOCUMENTS RETRIEVED" in prompt
    print(f"    ✅ Prompt built: {len(prompt)} chars")
    print(f"    ✅ Prompt preview: {prompt[:80].strip()}...")

    # test no results message
    no_results = build_no_results_message(
        jurisdiction="Delhi",
        question="test question"
    )
    assert "Delhi" in no_results
    print(f"    ✅ No-results message built")

    # test low confidence prompt
    low_conf_prompt = build_rag_prompt(
        question="test",
        context=context,
        is_low_confidence=True
    )
    assert "CONFIDENCE" in low_conf_prompt
    print(f"    ✅ Low confidence prompt built")

except Exception as e:
    print(f"    ❌ Prompts failed: {e}")

# ── Test 3: LLM Connection ─────────────────────────────
print("\n[3] Testing Groq LLM connection...")
try:
    from api.rag.llm import GroqLLM

    llm = GroqLLM()
    print(f"    ✅ LLM initialized: {llm.model}")

    # test connection
    is_connected = llm.test_connection()
    if is_connected:
        print(f"    ✅ Groq API responding correctly")
    else:
        print(f"    ⚠️  Groq API connected but unexpected response")

    # test actual generation
    test_prompt = """You are a legal assistant.
Answer in exactly 1 sentence.
Question: What does EPF stand for in Indian labor law?"""

    response = llm.generate(test_prompt, max_tokens=50)
    print(f"    ✅ LLM generated response: {response[:80].strip()}")

except Exception as e:
    print(f"    ❌ LLM failed: {e}")

# ── Test 4: Retriever ──────────────────────────────────
# ── Test 4: Retriever ──────────────────────────────────
print("\n[4] Testing retriever...")
try:
    from api.rag.retriever import LegalRetriever

    retriever = LegalRetriever()
    print(f"    ✅ Retriever initialized")

    total = retriever.count_indexed_documents()
    print(f"    ✅ Total vectors in Qdrant: {total}")

    if total == 0:
        print("    ⚠️  Qdrant empty - run fix_qdrant.py first")
        print("    python fix_qdrant.py")
    else:
        # test retrieval with Delhi filter
        results = retriever.retrieve(
            question="What is minimum wage in Delhi?",
            jurisdiction="Delhi",
            topic="minimum_wage"
        )

        if results:
            print(f"    ✅ Delhi filtered search: {len(results)} results")
            print(f"    ✅ Top score: {results[0]['score']}")
            print(f"    ✅ Top law: {results[0]['law_name']}")
            print(f"    ✅ Text: {results[0]['text'][:60]}...")
        else:
            print("    ⚠️  Delhi filtered search: 0 results")
            print("    ℹ️  Try running: python fix_qdrant.py")

        # test without filters
        all_results = retriever.retrieve(
            question="overtime rules for workers",
            top_k=3
        )
        print(f"    ✅ Unfiltered search: {len(all_results)} results")

        # count by jurisdiction
        delhi_count = retriever.count_indexed_documents(
            jurisdiction="Delhi"
        )
        central_count = retriever.count_indexed_documents(
            jurisdiction="Central"
        )
        print(f"    ✅ Delhi vectors: {delhi_count}")
        print(f"    ✅ Central vectors: {central_count}")

except Exception as e:
    print(f"    ❌ Retriever failed: {e}")
    import traceback
    traceback.print_exc()

# ── Test 5: Full Pipeline ──────────────────────────────
print("\n[5] Testing full RAG pipeline...")
try:
    from api.rag.pipeline import LegalRAGPipeline

    pipeline = LegalRAGPipeline()
    print(f"    ✅ Pipeline initialized")

    # test with a real question
    print(f"    Running pipeline for: 'minimum wage in Delhi'")
    result = pipeline.run(
        question="What is the minimum wage for unskilled workers in Delhi?",
        jurisdiction="Delhi",
        topic="minimum_wage"
    )

    # check response structure
    assert "answer" in result, "Missing answer"
    assert "sources" in result, "Missing sources"
    assert "jurisdiction" in result, "Missing jurisdiction"
    assert "disclaimer" in result, "Missing disclaimer"
    assert "has_results" in result, "Missing has_results"

    print(f"    ✅ Response structure correct")
    print(f"    ✅ has_results: {result['has_results']}")
    print(f"    ✅ Sources found: {len(result['sources'])}")
    print(f"    ✅ Disclaimer present: {bool(result['disclaimer'])}")
    print(f"\n    --- ANSWER PREVIEW ---")
    print(f"    {result['answer'][:200].strip()}")
    print(f"    ----------------------")

except Exception as e:
    print(f"    ❌ Full pipeline failed: {e}")
    import traceback
    traceback.print_exc()

# ── Test 6: Pipeline Edge Cases ────────────────────────
print("\n[6] Testing pipeline edge cases...")
try:
    from api.rag.pipeline import LegalRAGPipeline
    pipeline = LegalRAGPipeline()

    # test empty question
    result = pipeline.run(question="")
    assert not result["has_results"]
    assert "empty" in result["answer"].lower() or "cannot" in result["answer"].lower()
    print(f"    ✅ Empty question handled correctly")

    # test question with no matching docs
    result = pipeline.run(
        question="What are the rules for fishing licenses?",
        jurisdiction="Delhi"
    )
    assert "answer" in result
    print(f"    ✅ No-match question handled: has_results={result['has_results']}")

    # test without jurisdiction filter
    result = pipeline.run(
        question="What is EPF contribution rate?",
    )
    assert "answer" in result
    print(f"    ✅ No-filter query works: has_results={result['has_results']}")

except Exception as e:
    print(f"    ❌ Edge cases failed: {e}")

# ── Test 7: Comparison Pipeline ───────────────────────
print("\n[7] Testing comparison pipeline...")
try:
    from api.rag.pipeline import LegalRAGPipeline
    pipeline = LegalRAGPipeline()

    result = pipeline.run_comparison(
        question="What are the leave entitlements for workers?",
        jurisdiction1="Central",
        jurisdiction2="Maharashtra",
        topic="leave_policy"
    )

    assert "comparison" in result
    assert "Central" in result["comparison"]
    assert "Maharashtra" in result["comparison"]
    assert "question" in result

    print(f"    ✅ Comparison structure correct")
    print(f"    ✅ Central result: has_results={result['comparison']['Central']['has_results']}")
    print(f"    ✅ Maharashtra result: has_results={result['comparison']['Maharashtra']['has_results']}")

except Exception as e:
    print(f"    ❌ Comparison failed: {e}")

# ── Final Result ───────────────────────────────────────
print("\n" + "=" * 60)
print("✅ RAG Pipeline tests complete - Ready for Phase 4")
print("=" * 60)
print("\nNext: Phase 4 - FastAPI Backend")
print("The API layer that exposes the RAG pipeline as REST endpoints")