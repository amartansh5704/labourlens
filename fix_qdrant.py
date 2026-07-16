# fix_qdrant.py
# Clears old test data and re-seeds with correct metadata
# Run this once to fix the Qdrant state

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dotenv import load_dotenv
load_dotenv()

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams
from scraper.indexer.embedder import Embedder
from scraper.indexer.qdrant_indexer import QdrantIndexer
from loguru import logger

COLLECTION = os.getenv("QDRANT_COLLECTION", "employment_law_india")
HOST = os.getenv("QDRANT_HOST", "localhost")
PORT = int(os.getenv("QDRANT_PORT", 6333))

print("=" * 55)
print("Qdrant Cleanup and Re-seed")
print("=" * 55)

# ── Step 1: Connect ────────────────────────────────────
print("\n[1] Connecting to Qdrant...")
try:
    client = QdrantClient(host=HOST, port=PORT)
    collections = client.get_collections()
    print(f"    ✅ Connected to Qdrant at {HOST}:{PORT}")
except Exception as e:
    print(f"    ❌ Cannot connect to Qdrant: {e}")
    print("    Run: docker compose up -d qdrant")
    sys.exit(1)

# ── Step 2: Delete existing collection ────────────────
print(f"\n[2] Deleting collection: {COLLECTION}")
try:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION in existing:
        client.delete_collection(COLLECTION)
        print(f"    ✅ Deleted collection: {COLLECTION}")
    else:
        print(f"    ℹ️  Collection did not exist yet")
except Exception as e:
    print(f"    ❌ Delete failed: {e}")
    sys.exit(1)

# ── Step 3: Recreate collection ───────────────────────
print(f"\n[3] Creating fresh collection: {COLLECTION}")
try:
    client.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(
            size=384,
            distance=Distance.COSINE
        )
    )
    print(f"    ✅ Fresh collection created")
except Exception as e:
    print(f"    ❌ Create failed: {e}")
    sys.exit(1)

# ── Step 4: Load embedder ─────────────────────────────
print(f"\n[4] Loading embedding model...")
try:
    embedder = Embedder()
    print(f"    ✅ Embedder ready")
except Exception as e:
    print(f"    ❌ Embedder failed: {e}")
    sys.exit(1)

# ── Step 5: Seed proper test data ─────────────────────
print(f"\n[5] Seeding test data with correct metadata...")

test_chunks = [
    {
        "chunk_id": "seed-001",
        "document_id": "doc-delhi-001",
        "text": (
            "The minimum wages for unskilled workers in Delhi "
            "shall be Rs. 17,494 per month with effect from "
            "October 2023 as notified by the Delhi government "
            "under the Minimum Wages Act 1948. This includes "
            "basic wage of Rs. 12,468 plus variable dearness "
            "allowance of Rs. 5,026."
        ),
        "chunk_index": 0,
        "jurisdiction": "Delhi",
        "topic": "minimum_wage",
        "law_name": "Delhi Minimum Wages Notification 2023",
        "document_type": "notification",
        "agency": "Delhi Labour Department",
        "source_url": "https://labour.delhi.gov.in/minimum-wages",
        "title": "Delhi Minimum Wages 2023",
        "effective_date": "October 2023",
        "file_type": "html",
    },
    {
        "chunk_id": "seed-002",
        "document_id": "doc-delhi-002",
        "text": (
            "Semi-skilled workers in Delhi are entitled to "
            "minimum wages of Rs. 19,279 per month. Skilled "
            "workers shall receive Rs. 21,215 per month. "
            "These rates are revised twice yearly in April "
            "and October by the Delhi Labour Department."
        ),
        "chunk_index": 1,
        "jurisdiction": "Delhi",
        "topic": "minimum_wage",
        "law_name": "Delhi Minimum Wages Notification 2023",
        "document_type": "notification",
        "agency": "Delhi Labour Department",
        "source_url": "https://labour.delhi.gov.in/minimum-wages",
        "title": "Delhi Minimum Wages 2023",
        "effective_date": "October 2023",
        "file_type": "html",
    },
    {
        "chunk_id": "seed-003",
        "document_id": "doc-central-001",
        "text": (
            "No adult worker shall be required or allowed "
            "to work in a factory for more than 9 hours in "
            "any day or 48 hours in any week. Any work "
            "beyond this limit shall be treated as overtime "
            "and must be paid at double the ordinary rate "
            "of wages for the period of overtime work."
        ),
        "chunk_index": 0,
        "jurisdiction": "Central",
        "topic": "working_hours",
        "law_name": "Factories Act 1948",
        "document_type": "statute",
        "agency": "Ministry of Labour and Employment",
        "source_url": "https://labour.gov.in/factories-act",
        "title": "Factories Act 1948 - Working Hours",
        "effective_date": "1948",
        "file_type": "html",
    },
    {
        "chunk_id": "seed-004",
        "document_id": "doc-central-002",
        "text": (
            "Every employer shall contribute to the "
            "Employees Provident Fund at the rate of "
            "12 percent of the basic wages plus dearness "
            "allowance payable to each employee. The "
            "employee shall also contribute an equal "
            "amount of 12 percent to the fund."
        ),
        "chunk_index": 0,
        "jurisdiction": "Central",
        "topic": "epf_esi",
        "law_name": "Employees Provident Fund Act 1952",
        "document_type": "statute",
        "agency": "Employees Provident Fund Organisation",
        "source_url": "https://epfindia.gov.in/epf-act",
        "title": "EPF Act 1952",
        "effective_date": "1952",
        "file_type": "html",
    },
    {
        "chunk_id": "seed-005",
        "document_id": "doc-maharashtra-001",
        "text": (
            "Every worker who has completed one year of "
            "continuous service in an establishment shall "
            "be entitled to annual leave with wages for "
            "a period of 15 days. Casual leave of 8 days "
            "and sick leave of 7 days are also provided "
            "under the Maharashtra Shops and Establishments Act."
        ),
        "chunk_index": 0,
        "jurisdiction": "Maharashtra",
        "topic": "leave_policy",
        "law_name": "Maharashtra Shops and Establishments Act 1948",
        "document_type": "statute",
        "agency": "Maharashtra Labour Department",
        "source_url": "https://mahakamgar.maharashtra.gov.in",
        "title": "Maharashtra SE Act - Leave Policy",
        "effective_date": "1948",
        "file_type": "html",
    },
    {
        "chunk_id": "seed-006",
        "document_id": "doc-central-003",
        "text": (
            "Contract workers engaged through a contractor "
            "shall be entitled to the same wages as regular "
            "workers performing similar or identical work. "
            "The principal employer is jointly responsible "
            "for ensuring payment of wages and compliance "
            "with the Contract Labour Regulation and "
            "Abolition Act 1970."
        ),
        "chunk_index": 0,
        "jurisdiction": "Central",
        "topic": "worker_classification",
        "law_name": "Contract Labour Regulation and Abolition Act 1970",
        "document_type": "statute",
        "agency": "Ministry of Labour and Employment",
        "source_url": "https://labour.gov.in/contract-labour",
        "title": "Contract Labour Act 1970",
        "effective_date": "1970",
        "file_type": "html",
    },
    {
        "chunk_id": "seed-007",
        "document_id": "doc-karnataka-001",
        "text": (
            "The minimum wages for workers in Karnataka "
            "for the year 2024 have been revised upward. "
            "Unskilled workers shall receive Rs. 15,580 "
            "per month. Semi-skilled workers shall receive "
            "Rs. 16,848 per month as per the Karnataka "
            "Minimum Wages notification dated January 2024."
        ),
        "chunk_index": 0,
        "jurisdiction": "Karnataka",
        "topic": "minimum_wage",
        "law_name": "Karnataka Minimum Wages Notification 2024",
        "document_type": "notification",
        "agency": "Karnataka Labour Department",
        "source_url": "https://labour.kar.nic.in/minimum-wages",
        "title": "Karnataka Minimum Wages 2024",
        "effective_date": "January 2024",
        "file_type": "html",
    },
    {
        "chunk_id": "seed-008",
        "document_id": "doc-central-004",
        "text": (
            "The ESI contribution rate for employers is "
            "3.25 percent of the wages paid to employees. "
            "The employee contribution rate is 0.75 percent "
            "of wages. Employees earning up to Rs. 21,000 "
            "per month are covered under the Employees "
            "State Insurance Act 1948."
        ),
        "chunk_index": 0,
        "jurisdiction": "Central",
        "topic": "epf_esi",
        "law_name": "Employees State Insurance Act 1948",
        "document_type": "statute",
        "agency": "Employees State Insurance Corporation",
        "source_url": "https://esic.gov.in/esi-act",
        "title": "ESI Act 1948",
        "effective_date": "1948",
        "file_type": "html",
    },
]

try:
    indexer = QdrantIndexer()
    count = indexer.index_chunks(test_chunks, embedder)
    print(f"    ✅ Seeded {count} chunks into Qdrant")
except Exception as e:
    print(f"    ❌ Seeding failed: {e}")
    sys.exit(1)

# ── Step 6: Verify ────────────────────────────────────
print(f"\n[6] Verifying seed data...")
try:
    total = client.count(
        collection_name=COLLECTION,
        exact=True
    ).count
    print(f"    ✅ Total vectors in Qdrant: {total}")

    # verify Delhi data exists
    from qdrant_client.models import Filter, FieldCondition, MatchValue

    delhi_count = client.count(
        collection_name=COLLECTION,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="jurisdiction",
                    match=MatchValue(value="Delhi")
                )
            ]
        ),
        exact=True
    ).count
    print(f"    ✅ Delhi vectors: {delhi_count}")

    central_count = client.count(
        collection_name=COLLECTION,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="jurisdiction",
                    match=MatchValue(value="Central")
                )
            ]
        ),
        exact=True
    ).count
    print(f"    ✅ Central vectors: {central_count}")

    maharashtra_count = client.count(
        collection_name=COLLECTION,
        count_filter=Filter(
            must=[
                FieldCondition(
                    key="jurisdiction",
                    match=MatchValue(value="Maharashtra")
                )
            ]
        ),
        exact=True
    ).count
    print(f"    ✅ Maharashtra vectors: {maharashtra_count}")

except Exception as e:
    print(f"    ❌ Verification failed: {e}")

print("\n" + "=" * 55)
print("✅ Qdrant ready with proper test data")
print("Now run: python test_rag.py")
print("=" * 55)