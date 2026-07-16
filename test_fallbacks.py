# test_fallbacks.py
# Tests all 3 tiers of the pipeline

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from api.rag.pipeline import LegalRAGPipeline

pipeline = LegalRAGPipeline()

print("=" * 60)
print("Testing 3-Tier Fallback Pipeline")
print("=" * 60)

tests = [
    {
        "label": "TIER 1 - Should use indexed docs",
        "question": "What is minimum wage in Delhi?",
        "jurisdiction": "Delhi",
        "topic": "minimum_wage",
        "expected_source": "indexed_documents"
    },
    {
        "label": "TIER 1/2 - Missing jurisdiction",
        "question": "What is minimum wage in Haryana?",
        "jurisdiction": None,
        "topic": "minimum_wage",
        "expected_source": "web_search or llm_knowledge"
    },
    {
        "label": "TIER 2/3 - Recent changes",
        "question": "What are the 4 new labour codes in India?",
        "jurisdiction": "Central",
        "topic": None,
        "expected_source": "web_search or llm_knowledge"
    },
    {
        "label": "TIER 3 - General knowledge",
        "question": "What is the probation period law in India?",
        "jurisdiction": None,
        "topic": None,
        "expected_source": "llm_knowledge"
    },
    {
        "label": "TIER 2/3 - Very specific recent",
        "question": "What is the ESI wage ceiling in 2025?",
        "jurisdiction": "Central",
        "topic": "epf_esi",
        "expected_source": "web_search or llm_knowledge"
    },
]

for test in tests:
    print(f"\n{'─' * 60}")
    print(f"Test: {test['label']}")
    print(f"Q: {test['question']}")

    result = pipeline.run(
        question=test["question"],
        jurisdiction=test["jurisdiction"],
        topic=test["topic"],
        top_k=3
    )

    answer_source = result.get("answer_source", "unknown")
    has_results = result.get("has_results", False)
    sources_count = len(result.get("sources", []))
    answer_preview = result.get("answer", "")[:120]

    source_emoji = {
        "indexed_documents": "🟢",
        "web_search": "🔵",
        "llm_knowledge": "🟡",
        "error": "🔴",
    }.get(answer_source, "⚪")

    print(f"Source: {source_emoji} {answer_source}")
    print(f"Has results: {has_results} | Sources: {sources_count}")
    print(f"Answer: {answer_preview}...")

print(f"\n{'=' * 60}")
print("Fallback test complete")
print("=" * 60)