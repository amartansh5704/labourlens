# test_database.py
# Run this to verify database is working correctly

from database.connection import init_db, get_db_session, check_db_connection, get_db_stats
from database.models import Document, ScrapeLog, IndexLog
from shared.constants import JURISDICTION_NAMES, TOPIC_KEYS
from loguru import logger
import os

print("=" * 55)
print("LaborLens - Database Tests")
print("=" * 55)

# ── Test 1: Initialize Database ────────────────────────
print("\n[1] Initializing database...")
try:
    init_db()
    print("    ✅ Tables created successfully")
    print(f"    ✅ Database file: laborlens.db")
except Exception as e:
    print(f"    ❌ Failed: {e}")

# ── Test 2: Connection Check ───────────────────────────
print("\n[2] Checking database connection...")
if check_db_connection():
    print("    ✅ Database connection working")
else:
    print("    ❌ Database connection failed")

# ── Test 3: Insert a Test Document ────────────────────
print("\n[3] Testing document insert...")
db = get_db_session()
try:
    test_doc = Document(
        url="https://labour.gov.in/test-page",
        title="Test Minimum Wages Act",
        raw_text="The minimum wages for workers shall be determined by the government. Workers are entitled to fair wages as per this act.",
        jurisdiction="Central",
        topic="minimum_wage",
        document_type="statute",
        law_name="Minimum Wages Act 1948",
        agency="Ministry of Labour",
        effective_date="1948-03-15",
        file_type="html"
    )
    db.add(test_doc)
    db.commit()
    db.refresh(test_doc)
    print(f"    ✅ Document inserted: ID = {test_doc.id[:8]}...")
    print(f"    ✅ Title: {test_doc.title}")
    saved_id = test_doc.id
except Exception as e:
    print(f"    ❌ Insert failed: {e}")
    db.rollback()
    saved_id = None
finally:
    db.close()

# ── Test 4: Read Back the Document ────────────────────
print("\n[4] Testing document read...")
db = get_db_session()
try:
    if saved_id:
        doc = db.query(Document).filter(
            Document.id == saved_id
        ).first()

        if doc:
            print(f"    ✅ Document found: {doc.title}")
            print(f"    ✅ Jurisdiction: {doc.jurisdiction}")
            print(f"    ✅ Topic: {doc.topic}")
            print(f"    ✅ Text preview: {doc.raw_text[:60]}...")
        else:
            print("    ❌ Document not found after insert")
except Exception as e:
    print(f"    ❌ Read failed: {e}")
finally:
    db.close()

# ── Test 5: Insert a Scrape Log ───────────────────────
print("\n[5] Testing scrape log insert...")
db = get_db_session()
try:
    log = ScrapeLog(
        url="https://labour.gov.in/test-page",
        spider_name="central_labor",
        jurisdiction="Central",
        status="success",
        http_status=200,
        chunks_created=5,
        text_length=1500
    )
    db.add(log)
    db.commit()
    print(f"    ✅ Scrape log inserted: status={log.status}")
except Exception as e:
    print(f"    ❌ Scrape log failed: {e}")
    db.rollback()
finally:
    db.close()

# ── Test 6: Query by Jurisdiction and Topic ───────────
print("\n[6] Testing filter queries...")
db = get_db_session()
try:
    # by jurisdiction
    central_docs = db.query(Document).filter(
        Document.jurisdiction == "Central"
    ).all()
    print(f"    ✅ Central docs: {len(central_docs)}")

    # by topic
    wage_docs = db.query(Document).filter(
        Document.topic == "minimum_wage"
    ).all()
    print(f"    ✅ Minimum wage docs: {len(wage_docs)}")

    # by indexed status
    unindexed = db.query(Document).filter(
        Document.is_indexed == False
    ).all()
    print(f"    ✅ Unindexed docs: {len(unindexed)}")

except Exception as e:
    print(f"    ❌ Filter query failed: {e}")
finally:
    db.close()

# ── Test 7: Update Document ───────────────────────────
print("\n[7] Testing document update...")
db = get_db_session()
try:
    if saved_id:
        doc = db.query(Document).filter(
            Document.id == saved_id
        ).first()

        if doc:
            doc.is_indexed = True
            doc.chunk_count = 5
            db.commit()
            print(f"    ✅ Document updated: is_indexed=True, chunks=5")
except Exception as e:
    print(f"    ❌ Update failed: {e}")
    db.rollback()
finally:
    db.close()

# ── Test 8: Database Stats ────────────────────────────
print("\n[8] Testing database stats...")
try:
    stats = get_db_stats()
    print(f"    ✅ Total documents:  {stats['total_documents']}")
    print(f"    ✅ Indexed docs:     {stats['indexed_documents']}")
    print(f"    ✅ Total chunks:     {stats['total_chunks']}")
    print(f"    ✅ By jurisdiction:  {stats['by_jurisdiction']}")
    print(f"    ✅ By topic:         {stats['by_topic']}")
except Exception as e:
    print(f"    ❌ Stats failed: {e}")

# ── Test 9: Clean Up Test Data ────────────────────────
print("\n[9] Cleaning up test data...")
db = get_db_session()
try:
    # delete test document
    if saved_id:
        doc = db.query(Document).filter(
            Document.id == saved_id
        ).first()
        if doc:
            db.delete(doc)

    # delete test scrape logs
    db.query(ScrapeLog).filter(
        ScrapeLog.url == "https://labour.gov.in/test-page"
    ).delete()

    db.commit()
    print("    ✅ Test data cleaned up")
except Exception as e:
    print(f"    ❌ Cleanup failed: {e}")
    db.rollback()
finally:
    db.close()

# ── Test 10: Verify DB File Exists ────────────────────
print("\n[10] Checking database file...")
if os.path.exists("laborlens.db"):
    size = os.path.getsize("laborlens.db")
    print(f"    ✅ laborlens.db exists")
    print(f"    ✅ File size: {size} bytes")
else:
    print("    ❌ laborlens.db file not found")

# ── Final Result ───────────────────────────────────────
print("\n" + "=" * 55)
print("✅ Database phase complete - Ready for Phase 2")
print("=" * 55)