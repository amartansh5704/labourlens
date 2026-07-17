# fix_qdrant.py
# Clears old test data and re-seeds with correct metadata
# Works with both local Docker and Qdrant Cloud

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
HOST       = os.getenv("QDRANT_HOST", "localhost")
PORT       = int(os.getenv("QDRANT_PORT", 6333))
API_KEY    = os.getenv("QDRANT_API_KEY", "")

print("=" * 55)
print("Qdrant Cleanup and Re-seed")
print("=" * 55)
print(f"Host:       {HOST}")
print(f"API Key:    {'SET' if API_KEY else 'NOT SET'}")
print(f"Collection: {COLLECTION}")
print()

# ── Step 1: Connect ────────────────────────────────────
print("[1] Connecting to Qdrant...")
try:
    # use cloud connection if API key is set
    if API_KEY and "cloud.qdrant.io" in HOST:
        client = QdrantClient(
            url=f"https://{HOST}",
            api_key=API_KEY,
            timeout=30,
            check_compatibility=False,
        )
        print(f"    Using Qdrant Cloud: https://{HOST}")
    else:
        client = QdrantClient(
            host=HOST,
            port=PORT,
            timeout=30,
        )
        print(f"    Using local Qdrant: {HOST}:{PORT}")

    # test connection
    collections = client.get_collections()
    existing_names = [c.name for c in collections.collections]
    print(f"    ✅ Connected successfully")
    print(f"    Existing collections: {existing_names}")

except Exception as e:
    print(f"    ❌ Cannot connect to Qdrant: {e}")
    print()
    print("    Check your .env file:")
    print("    QDRANT_HOST=xxxx.eu-central-1.aws.cloud.qdrant.io")
    print("    QDRANT_API_KEY=your-api-key")
    print()
    print("    Or for local Docker:")
    print("    docker compose up -d qdrant")
    sys.exit(1)

# ── Step 2: Delete existing collection ────────────────
print(f"\n[2] Deleting collection: {COLLECTION}")
try:
    if COLLECTION in existing_names:
        client.delete_collection(COLLECTION)
        print(f"    ✅ Deleted: {COLLECTION}")
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
        "chunk_id": "seed-003",
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
        "chunk_id": "seed-004",
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
        "chunk_id": "seed-005",
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

    from qdrant_client.models import Filter, FieldCondition, MatchValue

    for jurisdiction in ["Delhi", "Central", "Maharashtra"]:
        j_count = client.count(
            collection_name=COLLECTION,
            count_filter=Filter(
                must=[
                    FieldCondition(
                        key="jurisdiction",
                        match=MatchValue(value=jurisdiction)
                    )
                ]
            ),
            exact=True
        ).count
        print(f"    ✅ {jurisdiction} vectors: {j_count}")

except Exception as e:
    print(f"    ❌ Verification failed: {e}")

print("\n" + "=" * 55)
print("✅ Qdrant ready with seed data")
print("Now run: python scraper/ingest_documents.py")
print("=" * 55)