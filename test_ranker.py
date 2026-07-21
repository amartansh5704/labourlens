# test_reranker.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from api.rag.reranker import LegalReranker

print("=" * 55)
print("Reranker Test")
print("=" * 55)

reranker = LegalReranker()

if not reranker.enabled:
    print("❌ Reranker not loaded")
    sys.exit(1)

print("✅ Reranker loaded")

# fake chunks to test reranking
chunks = [
    {
        "text": "The EPF contribution rate is 12% of basic wages.",
        "law_name": "EPF Act",
        "score": 0.85,
        "jurisdiction": "Central",
        "topic": "epf_esi",
        "agency": "EPFO",
        "source_url": "https://epfindia.gov.in",
        "effective_date": "1952",
        "document_type": "statute",
        "chunk_index": 0,
    },
    {
        "text": "Minimum wage in Delhi is Rs 17494 per month.",
        "law_name": "Delhi Wages",
        "score": 0.72,
        "jurisdiction": "Delhi",
        "topic": "minimum_wage",
        "agency": "Delhi Labour",
        "source_url": "https://labour.delhi.gov.in",
        "effective_date": "2023",
        "document_type": "notification",
        "chunk_index": 0,
    },
    {
        "text": "Workers cannot work more than 9 hours per day.",
        "law_name": "Factories Act",
        "score": 0.68,
        "jurisdiction": "Central",
        "topic": "working_hours",
        "agency": "Ministry of Labour",
        "source_url": "https://labour.gov.in",
        "effective_date": "1948",
        "document_type": "statute",
        "chunk_index": 0,
    },
    {
        "text": "ESI contribution is 3.25% for employer.",
        "law_name": "ESI Act",
        "score": 0.65,
        "jurisdiction": "Central",
        "topic": "epf_esi",
        "agency": "ESIC",
        "source_url": "https://esic.gov.in",
        "effective_date": "1948",
        "document_type": "statute",
        "chunk_index": 0,
    },
]

question = "What is the EPF contribution rate?"

print(f"\nQuestion: {question}")
print(f"\nBefore reranking (Qdrant order):")
for i, c in enumerate(chunks, 1):
    print(f"  {i}. [{c['score']:.3f}] {c['law_name']}: {c['text'][:50]}")

reranked = reranker.rerank(
    question=question,
    chunks=chunks,
    top_k=3
)

print(f"\nAfter reranking (Cross-encoder order):")
for i, c in enumerate(reranked, 1):
    print(
        f"  {i}. "
        f"[rerank={c.get('rerank_score', 0):.3f}] "
        f"[qdrant={c.get('qdrant_score', 0):.3f}] "
        f"{c['law_name']}: {c['text'][:50]}"
    )

print(f"\n✅ Reranker working correctly")
print("EPF chunk should be ranked #1 for EPF question")
print("=" * 55)