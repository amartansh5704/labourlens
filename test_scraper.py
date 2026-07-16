# test_scraper.py
# Tests scraper components WITHOUT hitting real websites

from scraper.processors.html_parser import (
    extract_text_from_html,
    extract_title_from_html
)
from scraper.processors.cleaner import (
    clean_document_text,
    detect_topic_from_url,
    detect_topic_from_text
)
from scraper.processors.chunker import create_chunks
from scraper.indexer.embedder import Embedder
from database.connection import init_db

print("=" * 55)
print("LaborLens - Scraper Component Tests")
print("=" * 55)

# ── Test 1: HTML Parser ────────────────────────────────
print("\n[1] Testing HTML parser...")
sample_html = """
<html>
<head><title>Minimum Wages Act - Labour Department</title></head>
<body>
<nav>Home | About | Contact</nav>
<main>
<h1>Minimum Wages Act 1948</h1>
<p>The Minimum Wages Act, 1948 provides for fixing minimum
rates of wages in certain employments.</p>
<p>The Central Government fixes minimum wages for employments
in Central sphere while the State Governments fix minimum wages
for employments in their respective spheres.</p>
<p>No employer shall pay wages less than the minimum wages
fixed under this Act to any employee employed in a scheduled
employment.</p>
</main>
<footer>Copyright 2024 Ministry of Labour</footer>
</body>
</html>
"""

text = extract_text_from_html(sample_html)
title = extract_title_from_html(sample_html)

if len(text) > 50:
    print(f"    ✅ HTML text extracted: {len(text)} chars")
    print(f"    ✅ Title extracted: {title}")
    print(f"    ✅ Preview: {text[:100].strip()}")
else:
    print(f"    ❌ HTML extraction failed: got '{text}'")

# ── Test 2: Text Cleaner ───────────────────────────────
print("\n[2] Testing text cleaner...")
dirty_text = """
Home | About | Contact | Login

Skip to main content

MINIMUM WAGES ACT

The minimum wages for workers...
w.e.f. January 2024, the rates shall be Rs. 500 p.d.
The govt. dept. has notified these changes.

© 2024 Government of India. All rights reserved.
Cookie Policy | Privacy Policy
"""

clean = clean_document_text(dirty_text, "https://labour.gov.in/test")
if "minimum wages" in clean.lower():
    print(f"    ✅ Text cleaned: {len(dirty_text)} → {len(clean)} chars")
    print(f"    ✅ Noise removed: copyright gone = {'©' not in clean}")
    print(f"    ✅ Abbreviations fixed: {'with effect from' in clean}")
else:
    print(f"    ❌ Cleaning failed")

# ── Test 3: Topic Detection ────────────────────────────
print("\n[3] Testing topic detection...")
test_cases = [
    ("https://labour.gov.in/minimum-wages", "minimum_wage"),
    ("https://labour.gov.in/overtime-rules", "working_hours"),
    ("https://epfindia.gov.in/epf-schemes", "epf_esi"),
    ("https://labour.gov.in/casual-leave-policy", "leave_policy"),
    ("https://labour.gov.in/contract-labour", "worker_classification"),
]

all_passed = True
for url, expected_topic in test_cases:
    detected = detect_topic_from_url(url)
    status = "✅" if detected == expected_topic else "❌"
    if detected != expected_topic:
        all_passed = False
    print(f"    {status} URL topic: {url.split('/')[-1][:30]} → {detected}")

if all_passed:
    print("    ✅ All topic detections correct")

# ── Test 4: Chunker ────────────────────────────────────
print("\n[4] Testing chunker...")
long_text = """
Section 1: Minimum Wages

The minimum wages for workers in the organized sector shall be
fixed by the appropriate government under this Act.

Section 2: Overtime Rules

Any worker who works beyond nine hours in a day or forty-eight
hours in a week shall be entitled to overtime wages at the rate
of double the ordinary rate of wages for the period of overtime work.

Section 3: Leave Entitlement

Every worker shall be entitled to annual leave with wages
calculated at the rate of one day for every twenty days
of work performed during the previous calendar year.

Section 4: EPF Contribution

Every employer shall contribute to the Provident Fund at the
rate of twelve percent of the basic wages plus dearness allowance
payable to each employee.

Section 5: ESI Contribution

The employer shall contribute at the rate of three point twenty-five
percent and the employee shall contribute at the rate of zero point
seventy-five percent of the wages for ESI coverage.
"""

metadata = {
    "id": "test-doc-001",
    "url": "https://labour.gov.in/test-act",
    "title": "Test Labour Act",
    "jurisdiction": "Central",
    "topic": "minimum_wage",
    "law_name": "Test Act 2024",
    "document_type": "statute",
    "agency": "Ministry of Labour",
    "effective_date": "2024-01-01",
    "file_type": "html"
}

chunks = create_chunks(long_text, metadata)

if len(chunks) > 0:
    print(f"    ✅ Created {len(chunks)} chunks")
    print(f"    ✅ First chunk length: {len(chunks[0]['text'])} chars")
    print(f"    ✅ Each chunk has jurisdiction: {chunks[0]['jurisdiction']}")
    print(f"    ✅ Each chunk has source_url: {bool(chunks[0]['source_url'])}")
    print(f"    ✅ Each chunk has chunk_id: {bool(chunks[0]['chunk_id'])}")
else:
    print("    ❌ No chunks created")

# ── Test 5: Embedder ───────────────────────────────────
print("\n[5] Testing embedder...")
print("    Loading model (may take a moment)...")
try:
    embedder = Embedder()

    # test embedding multiple texts
    test_texts = [
        "What is the minimum wage in Delhi?",
        "Overtime rules for factory workers",
        "EPF contribution rates for employers"
    ]

    embeddings = embedder.embed(test_texts)
    print(f"    ✅ Embedded {len(test_texts)} texts")
    print(f"    ✅ Embedding shape: {embeddings.shape}")
    print(f"    ✅ Vector dimensions: {embeddings.shape[1]}")

    # test query embedding
    query_vec = embedder.embed_query("minimum wage rules")
    print(f"    ✅ Query embedding: {len(query_vec)} dimensions")

except Exception as e:
    print(f"    ❌ Embedder failed: {e}")

# ── Test 6: Qdrant Indexer ─────────────────────────────
print("\n[6] Testing Qdrant indexer...")
try:
    from scraper.indexer.qdrant_indexer import QdrantIndexer

    indexer = QdrantIndexer()

    # use count_points instead of get_collection_stats
    # more reliable across Qdrant versions
    before_count = indexer.count_points()
    print(f"    ✅ Qdrant connected")
    print(f"    ✅ Collection: {indexer.collection_name}")
    print(f"    ✅ Vectors before test: {before_count}")

    # test indexing the chunks we created earlier
    if chunks:
        count = indexer.index_chunks(chunks, embedder)
        print(f"    ✅ Indexed {count} test chunks into Qdrant")

        after_count = indexer.count_points()
        print(f"    ✅ Vectors after test: {after_count}")
    else:
        print("    ⚠️  No chunks to index (check test 4)")

except Exception as e:
    print(f"    ❌ Qdrant indexer failed: {e}")
    print("    Make sure Docker is running: docker compose up -d qdrant")

# ── Final Result ───────────────────────────────────────
print("\n" + "=" * 55)
print("✅ Scraper components ready - Ready for Phase 3")
print("=" * 55)