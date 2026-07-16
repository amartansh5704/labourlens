# test_real_data.py
# Tests RAG pipeline against real indexed documents
# Run after ingest_documents.py completes

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from api.rag.pipeline import LegalRAGPipeline
from api.rag.retriever import LegalRetriever

print("=" * 65)
print("LaborLens - Real Data Quality Test")
print("=" * 65)

# ── Check what is in Qdrant ────────────────────────────
print("\n[1] Checking indexed data...")
retriever = LegalRetriever()

total = retriever.count_indexed_documents()
print(f"    Total vectors in Qdrant: {total}")

if total < 10:
    print("\n    ⚠️  Very few documents indexed")
    print("    Run these first:")
    print("    1. python scraper/ingest_urls.py")
    print("    2. python scraper/ingest_documents.py")

# count per jurisdiction
from shared.constants import JURISDICTION_NAMES, TOPIC_KEYS
print("\n    Vectors by jurisdiction:")
for j in JURISDICTION_NAMES:
    count = retriever.count_indexed_documents(jurisdiction=j)
    bar = "█" * min(count, 30)
    print(f"    {j:<15} {count:>4} {bar}")

print("\n    Vectors by topic:")
for t in TOPIC_KEYS:
    count = retriever.count_indexed_documents(topic=t)
    bar = "█" * min(count, 30)
    print(f"    {t:<25} {count:>4} {bar}")

# ── Real Questions Test ────────────────────────────────
print("\n" + "=" * 65)
print("[2] Testing with REAL questions...")
print("=" * 65)

pipeline = LegalRAGPipeline()

real_questions = [
    {
        "question": "What is the minimum wage for unskilled workers?",
        "jurisdiction": "Delhi",
        "topic": "minimum_wage",
        "label": "Delhi Minimum Wage"
    },
    {
        "question": "What is the minimum wage rate in Karnataka for 2026?",
        "jurisdiction": "Karnataka",
        "topic": "minimum_wage",
        "label": "Karnataka Minimum Wage 2026"
    },
    {
        "question": "What are the working hours and overtime rules?",
        "jurisdiction": "Central",
        "topic": "working_hours",
        "label": "Factory Working Hours"
    },
    {
        "question": "What is the EPF contribution rate for employers and employees?",
        "jurisdiction": "Central",
        "topic": "epf_esi",
        "label": "EPF Contribution Rate"
    },
    {
        "question": "What is the ESI contribution rate?",
        "jurisdiction": "Central",
        "topic": "epf_esi",
        "label": "ESI Contribution Rate"
    },
    {
        "question": "What are the maternity leave entitlements?",
        "jurisdiction": "Central",
        "topic": "leave_policy",
        "label": "Maternity Leave"
    },
    {
        "question": "What are the minimum wage rates in Tamil Nadu?",
        "jurisdiction": "Tamil Nadu",
        "topic": "minimum_wage",
        "label": "Tamil Nadu Minimum Wage"
    },
    {
        "question": "What are the minimum wages in Telangana?",
        "jurisdiction": "Telangana",
        "topic": "minimum_wage",
        "label": "Telangana Minimum Wage"
    },
    {
        "question": "What rights do contract workers have?",
        "jurisdiction": "Central",
        "topic": "worker_classification",
        "label": "Contract Worker Rights"
    },
    {
        "question": "What are leave entitlements under Maharashtra shops act?",
        "jurisdiction": "Maharashtra",
        "topic": "leave_policy",
        "label": "Maharashtra Leave Policy"
    },
]

passed = 0
failed = 0

for i, q in enumerate(real_questions, 1):
    print(f"\n[Q{i}] {q['label']}")
    print(f"     Question: {q['question']}")
    print(f"     Filter:   {q['jurisdiction']} | {q['topic']}")

    result = pipeline.run(
        question=q["question"],
        jurisdiction=q["jurisdiction"],
        topic=q["topic"],
        top_k=3
    )

    if result["has_results"]:
        passed += 1
        top_score = result.get("top_score", 0)
        sources_count = len(result["sources"])

        print(f"     ✅ ANSWERED | Sources: {sources_count} | "
              f"Top score: {top_score:.3f}")

        # show answer preview
        answer_lines = result["answer"].strip().split("\n")
        for line in answer_lines[:4]:
            if line.strip():
                print(f"     📝 {line.strip()[:80]}")

        # show top source
        if result["sources"]:
            src = result["sources"][0]
            print(f"     📚 Source: {src['law_name']}")

    else:
        failed += 1
        print(f"     ❌ NO RESULTS for this question")
        print(f"        Either data not indexed or topic mismatch")

# ── Summary ────────────────────────────────────────────
print("\n" + "=" * 65)
print("REAL DATA TEST SUMMARY")
print("=" * 65)
print(f"✅ Answered:    {passed}/{len(real_questions)} questions")
print(f"❌ No results: {failed}/{len(real_questions)} questions")
print(f"📊 Coverage:   {round(passed/len(real_questions)*100)}%")

if passed >= 7:
    print("\n🎉 GREAT! Data quality is good")
    print("Ready to build the frontend!")
elif passed >= 4:
    print("\n⚠️  DECENT coverage but add more documents")
    print("Missing jurisdictions need more PDFs")
else:
    print("\n❌ LOW coverage - need more documents")
    print("Run ingest_urls.py and add more PDFs")
print("=" * 65)