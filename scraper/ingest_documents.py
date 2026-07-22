# scraper/ingest_documents.py
# Reads all PDFs and text files from documents/ folder
# Extracts text, detects metadata from path
# Chunks and indexes everything into Qdrant
# Use --force flag to re-index everything to cloud

import sys
import os
import argparse
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from dotenv import load_dotenv
load_dotenv()

import pdfplumber
import ftfy
import re
from pathlib import Path
from loguru import logger
from database.connection import init_db, get_db_session
from database.models import Document, IndexLog
from scraper.processors.chunker import create_chunks
from scraper.indexer.embedder import Embedder
from scraper.indexer.qdrant_indexer import QdrantIndexer
from datetime import datetime
from typing import Optional
import uuid

# ─────────────────────────────────────────────────────────
# METADATA MAPS
# ─────────────────────────────────────────────────────────
JURISDICTION_MAP = {
    "central": "Central",
    "delhi": "Delhi",
    "maharashtra": "Maharashtra",
    "karnataka": "Karnataka",
    "tamil_nadu": "Tamil Nadu",
    "telangana": "Telangana",
}

TOPIC_MAP = {
    "minimum_wage": "minimum_wage",
    "working_hours": "working_hours",
    "epf_esi": "epf_esi",
    "leave_policy": "leave_policy",
    "worker_classification": "worker_classification",
}

AGENCY_MAP = {
    "central": "Ministry of Labour and Employment, India",
    "delhi": "Delhi Labour Department",
    "maharashtra": "Maharashtra Labour Department",
    "karnataka": "Karnataka Labour Department",
    "tamil_nadu": "Tamil Nadu Labour Department",
    "telangana": "Telangana Labour Department",
}

DOCUMENTS_ROOT = Path("documents")


# ─────────────────────────────────────────────────────────
# METADATA DETECTION
# ─────────────────────────────────────────────────────────
def detect_metadata_from_path(file_path: Path) -> dict:
    """Detect jurisdiction and topic from folder structure"""
    parts = file_path.parts

    try:
        doc_idx = list(parts).index("documents")
        relative_parts = parts[doc_idx + 1:]
    except ValueError:
        relative_parts = parts

    jurisdiction_key = (
        relative_parts[0] if len(relative_parts) > 0
        else "central"
    )
    topic_key = (
        relative_parts[1] if len(relative_parts) > 1
        else "general"
    )
    filename = file_path.stem
    law_name = (
        filename.replace("_", " ").replace("-", " ").title()
    )

    return {
        "jurisdiction": JURISDICTION_MAP.get(
            jurisdiction_key, jurisdiction_key.title()
        ),
        "topic": TOPIC_MAP.get(topic_key, topic_key),
        "agency": AGENCY_MAP.get(
            jurisdiction_key, "Government of India"
        ),
        "law_name": law_name,
        "file_type": file_path.suffix.lower().strip("."),
    }


# ─────────────────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────────────────
def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text - pdfplumber only (PyMuPDF removed)"""

    try:
        text_parts = []
        with pdfplumber.open(str(file_path)) as pdf:
            logger.info(f"PDF has {len(pdf.pages)} pages")
            for i, page in enumerate(pdf.pages):
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    text_parts.append(f"\n--- Page {i+1} ---\n")
                    text_parts.append(page_text)

        text = "\n".join(text_parts)
        text = ftfy.fix_text(text)

        if len(text.strip()) > 200:
            logger.info(
                f"pdfplumber extracted {len(text):,} chars"
            )
            return clean_text(text)

    except Exception as e:
        logger.warning(f"pdfplumber failed: {e}")

    logger.error(f"Could not extract text from: {file_path}")
    return ""


def extract_text_from_txt(file_path: Path) -> tuple:
    """
    Extract text from .txt files.
    Returns (text, extra_metadata_dict)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        extra_metadata = {}
        text = content

        if "---CONTENT_START---" in content:
            header, text = content.split(
                "---CONTENT_START---", 1
            )
            for line in header.strip().split("\n"):
                if ":" in line:
                    key, value = line.split(":", 1)
                    key = key.strip().upper()
                    value = value.strip()
                    if key == "SOURCE_URL":
                        extra_metadata["url"] = value
                    elif key == "LAW_NAME":
                        extra_metadata["law_name"] = value
                    elif key == "JURISDICTION":
                        extra_metadata["jurisdiction"] = value
                    elif key == "TOPIC":
                        extra_metadata["topic"] = value
                    elif key == "AGENCY":
                        extra_metadata["agency"] = value

        text = ftfy.fix_text(text.strip())
        text = clean_text(text)
        return text, extra_metadata

    except Exception as e:
        logger.error(f"Failed to read txt file: {e}")
        return "", {}


def clean_text(text: str) -> str:
    """Clean extracted text"""
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"^\.{3,}.*$", "", text, flags=re.MULTILINE)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r" {2,}", " ", text)
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            lines.append("")
            continue
        if re.match(r"^[\d\s\W]{1,3}$", stripped):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────
# DOCUMENT DISCOVERY
# ─────────────────────────────────────────────────────────
def discover_documents() -> list:
    """Find all PDF and TXT files in documents/ folder"""
    if not DOCUMENTS_ROOT.exists():
        logger.error(
            f"Documents folder not found: {DOCUMENTS_ROOT}"
        )
        return []

    files = []
    for pdf in DOCUMENTS_ROOT.rglob("*.pdf"):
        files.append(pdf)
    for txt in DOCUMENTS_ROOT.rglob("*.txt"):
        files.append(txt)

    logger.info(f"Found {len(files)} files in documents/")
    return sorted(files)


# ─────────────────────────────────────────────────────────
# FIND OR CREATE DOCUMENT IN SQLITE
# ─────────────────────────────────────────────────────────
def get_or_create_document(
    db,
    file_url: str,
    source_url: str,
    text: str,
    metadata: dict,
) -> Optional[Document]:
    """
    Finds existing document by URL (tries both file_url
    and source_url) or creates a new one.
    Never causes UNIQUE constraint errors.
    """

    # try source_url first (web URLs from txt headers)
    doc = None
    if source_url and source_url != file_url:
        doc = db.query(Document).filter(
            Document.url == source_url
        ).first()

    # try file_url if source_url not found
    if not doc:
        doc = db.query(Document).filter(
            Document.url == file_url
        ).first()

    if doc:
        # UPDATE existing record
        doc.raw_text = text
        doc.title = metadata["law_name"]
        doc.jurisdiction = metadata["jurisdiction"]
        doc.topic = metadata["topic"]
        doc.law_name = metadata["law_name"]
        doc.agency = metadata["agency"]
        doc.is_indexed = False
        doc.chunk_count = 0
        db.commit()
        return doc, "updated"

    else:
        # CREATE new record
        # use source_url if available otherwise file_url
        url_to_store = source_url if source_url else file_url

        new_doc = Document(
            id=str(uuid.uuid4()),
            url=url_to_store,
            title=metadata["law_name"],
            raw_text=text,
            jurisdiction=metadata["jurisdiction"],
            topic=metadata["topic"],
            document_type="statute",
            law_name=metadata["law_name"],
            agency=metadata["agency"],
            effective_date="",
            file_type=metadata["file_type"],
            scraped_at=datetime.utcnow(),
            is_indexed=False,
            chunk_count=0
        )
        db.add(new_doc)
        db.commit()
        db.refresh(new_doc)
        return new_doc, "created"


# ─────────────────────────────────────────────────────────
# MAIN INGESTION
# ─────────────────────────────────────────────────────────
def ingest_all_documents(force_reindex: bool = False):
    """
    Process all files in documents/ folder.

    Args:
        force_reindex: re-indexes ALL documents even if
                       already marked as indexed.
                       Use when switching to cloud Qdrant.
    """

    print("=" * 60)
    print("LaborLens - Document Ingestion Pipeline")
    if force_reindex:
        print("MODE: FORCE RE-INDEX (pushing to cloud Qdrant)")
    print("=" * 60)

    init_db()
    files = discover_documents()

    if not files:
        print("\n❌ No files found in documents/ folder")
        return

    print(f"\nFound {len(files)} files to process\n")

    logger.info("Loading embedding model...")
    embedder = Embedder()
    indexer = QdrantIndexer()

    qdrant_host = os.getenv("QDRANT_HOST", "localhost")
    if "cloud.qdrant.io" in qdrant_host:
        print(f"📡 Indexing to CLOUD Qdrant: {qdrant_host}")
    else:
        print(f"🖥️  Indexing to LOCAL Qdrant: {qdrant_host}")

    db = get_db_session()

    stats = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "total_chunks": 0,
    }

    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] {file_path.name}")
        print(f"            {file_path}")

        file_url = f"file://{file_path.resolve()}"

        # check skip condition (only when not force)
        if not force_reindex:
            existing = db.query(Document).filter(
                Document.url == file_url
            ).first()
            if existing and existing.is_indexed:
                print(f"    ⏭️  Already indexed, skipping")
                print(
                    "    💡 Use --force to re-index to cloud"
                )
                stats["skipped"] += 1
                continue

        # detect metadata from path
        metadata = detect_metadata_from_path(file_path)
        print(
            f"    📁 {metadata['jurisdiction']} | "
            f"{metadata['topic']}"
        )

        # extract text
        extra_metadata = {}

        if file_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)

        elif file_path.suffix.lower() == ".txt":
            text, extra_metadata = extract_text_from_txt(
                file_path
            )
            # override path metadata with file header metadata
            if extra_metadata.get("jurisdiction"):
                metadata["jurisdiction"] = (
                    extra_metadata["jurisdiction"]
                )
            if extra_metadata.get("topic"):
                metadata["topic"] = extra_metadata["topic"]
            if extra_metadata.get("law_name"):
                metadata["law_name"] = (
                    extra_metadata["law_name"]
                )
            if extra_metadata.get("agency"):
                metadata["agency"] = extra_metadata["agency"]
        else:
            print(f"    ⚠️  Unsupported: {file_path.suffix}")
            stats["failed"] += 1
            continue

        if not text or len(text) < 100:
            print(
                f"    ❌ Too little text: {len(text)} chars"
            )
            stats["failed"] += 1
            continue

        print(f"    📝 {len(text):,} characters extracted")

        # save or update in SQLite - NO UNIQUE errors
        try:
            source_url = extra_metadata.get("url", file_url)

            doc, action = get_or_create_document(
                db=db,
                file_url=file_url,
                source_url=source_url,
                text=text,
                metadata=metadata,
            )
            print(f"    💾 {action.title()} in database")

        except Exception as e:
            logger.error(f"Database operation failed: {e}")
            db.rollback()
            stats["failed"] += 1
            continue

        # create chunks
        chunks = create_chunks(
            text=text,
            document_metadata={
                "id": doc.id,
                "url": doc.url,
                "title": doc.title or "",
                "jurisdiction": doc.jurisdiction,
                "topic": doc.topic,
                "law_name": doc.law_name or "",
                "document_type": doc.document_type or "",
                "agency": doc.agency or "",
                "effective_date": doc.effective_date or "",
                "file_type": doc.file_type or "html",
            }
        )

        if not chunks:
            print(f"    ❌ No chunks created")
            stats["failed"] += 1
            continue

        print(f"    🔪 {len(chunks)} chunks created")

        # index into Qdrant
        try:
            chunk_count = indexer.index_chunks(
                chunks=chunks,
                embedder=embedder
            )

            doc.is_indexed = True
            doc.chunk_count = chunk_count
            db.commit()

            log = IndexLog(
                document_id=doc.id,
                document_url=doc.url,
                status="success",
                chunks_indexed=chunk_count,
                indexed_at=datetime.utcnow()
            )
            db.add(log)
            db.commit()

            stats["processed"] += 1
            stats["total_chunks"] += chunk_count
            print(
                f"    ✅ {chunk_count} chunks indexed to Qdrant"
            )

        except Exception as e:
            logger.error(f"Qdrant indexing failed: {e}")
            stats["failed"] += 1
            continue

    db.close()

    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"✅ Processed:    {stats['processed']} documents")
    print(f"⏭️  Skipped:      {stats['skipped']} documents")
    print(f"❌ Failed:       {stats['failed']} documents")
    print(f"🔢 Total chunks: {stats['total_chunks']}")

    try:
        total_vectors = indexer.count_points()
        print(f"📊 Qdrant total: {total_vectors} vectors")
        if "cloud.qdrant.io" in qdrant_host:
            print(f"📡 Cloud Qdrant: {qdrant_host}")
    except Exception:
        pass

    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Ingest documents into LaborLens"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-index all documents to cloud Qdrant"
    )
    args = parser.parse_args()
    ingest_all_documents(force_reindex=args.force)