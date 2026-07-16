# scraper/ingest_documents.py
# Reads all PDFs and text files from documents/ folder
# Extracts text, detects metadata from path
# Chunks and indexes everything into Qdrant

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import pdfplumber
import fitz  # PyMuPDF
import ftfy
import re
import io
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
# METADATA MAP
# Maps folder names to proper values
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

# documents folder path
DOCUMENTS_ROOT = Path("documents")


# ─────────────────────────────────────────────────────────
# METADATA DETECTION
# ─────────────────────────────────────────────────────────
def detect_metadata_from_path(file_path: Path) -> dict:
    """
    Detect jurisdiction, topic, law name from file path.

    Path structure:
    documents/[jurisdiction]/[topic]/[filename]

    Example:
    documents/delhi/minimum_wage/delhi_wages_2024.pdf
    → jurisdiction=Delhi, topic=minimum_wage
    """
    parts = file_path.parts

    # find documents/ in path and get parts after it
    try:
        doc_idx = list(parts).index("documents")
        relative_parts = parts[doc_idx + 1:]
    except ValueError:
        relative_parts = parts

    jurisdiction_key = relative_parts[0] if len(relative_parts) > 0 else "central"
    topic_key = relative_parts[1] if len(relative_parts) > 1 else "general"
    filename = file_path.stem  # filename without extension

    # clean filename to make law name
    law_name = filename.replace("_", " ").replace("-", " ").title()

    return {
        "jurisdiction": JURISDICTION_MAP.get(
            jurisdiction_key, jurisdiction_key.title()
        ),
        "topic": TOPIC_MAP.get(topic_key, topic_key),
        "agency": AGENCY_MAP.get(
            jurisdiction_key,
            "Government of India"
        ),
        "law_name": law_name,
        "file_type": file_path.suffix.lower().strip("."),
    }


# ─────────────────────────────────────────────────────────
# TEXT EXTRACTION
# ─────────────────────────────────────────────────────────
def extract_text_from_pdf(file_path: Path) -> str:
    """Extract text from PDF using pdfplumber then PyMuPDF"""

    # try pdfplumber first
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

    # fallback to PyMuPDF
    try:
        text_parts = []
        doc = fitz.open(str(file_path))
        logger.info(f"PyMuPDF: PDF has {len(doc)} pages")

        for i, page in enumerate(doc):
            page_text = page.get_text()
            if page_text and page_text.strip():
                text_parts.append(f"\n--- Page {i+1} ---\n")
                text_parts.append(page_text)

        doc.close()
        text = "\n".join(text_parts)
        text = ftfy.fix_text(text)

        if len(text.strip()) > 200:
            logger.info(
                f"PyMuPDF extracted {len(text):,} chars"
            )
            return clean_text(text)

    except Exception as e:
        logger.warning(f"PyMuPDF also failed: {e}")

    logger.error(f"Could not extract text from: {file_path}")
    return ""


def extract_text_from_txt(file_path: Path) -> tuple:
    """
    Extract text from .txt files saved by URL scraper.
    Also extracts metadata from the header we added.

    Returns (text, extra_metadata_dict)
    """
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        extra_metadata = {}
        text = content

        # check if file has our metadata header
        if "---CONTENT_START---" in content:
            header, text = content.split("---CONTENT_START---", 1)

            # parse metadata from header
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
    # fix hyphenation
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # normalize whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # remove lines with just dots (TOC)
    text = re.sub(r"^\.{3,}.*$", "", text, flags=re.MULTILINE)

    # collapse multiple blank lines
    text = re.sub(r"\n{3,}", "\n\n", text)

    # collapse multiple spaces
    text = re.sub(r" {2,}", " ", text)

    # remove very short lines that are likely artifacts
    lines = []
    for line in text.split("\n"):
        stripped = line.strip()
        # keep empty lines for paragraph breaks
        if not stripped:
            lines.append("")
            continue
        # skip lines that are just numbers or single chars
        if re.match(r"^[\d\s\W]{1,3}$", stripped):
            continue
        lines.append(line)

    return "\n".join(lines).strip()


# ─────────────────────────────────────────────────────────
# DOCUMENT DISCOVERY
# ─────────────────────────────────────────────────────────
def discover_documents() -> list:
    """
    Find all PDF and TXT files in documents/ folder.
    Returns list of file paths.
    """
    if not DOCUMENTS_ROOT.exists():
        logger.error(
            f"Documents folder not found: {DOCUMENTS_ROOT}"
        )
        return []

    files = []

    # find all PDFs
    for pdf in DOCUMENTS_ROOT.rglob("*.pdf"):
        files.append(pdf)

    # find all TXT files (scraped from URLs)
    for txt in DOCUMENTS_ROOT.rglob("*.txt"):
        files.append(txt)

    logger.info(f"Found {len(files)} files in documents/")
    return sorted(files)


# ─────────────────────────────────────────────────────────
# MAIN INGESTION
# ─────────────────────────────────────────────────────────
def ingest_all_documents():
    """
    Main function that:
    1. Discovers all files in documents/
    2. Extracts text from each
    3. Saves to SQLite
    4. Creates chunks
    5. Indexes into Qdrant
    """

    print("=" * 60)
    print("LaborLens - Document Ingestion Pipeline")
    print("=" * 60)

    # init database
    init_db()

    # discover files
    files = discover_documents()

    if not files:
        print("\n❌ No files found in documents/ folder")
        print("Add PDFs or run: python scraper/ingest_urls.py")
        return

    print(f"\nFound {len(files)} files to process\n")

    # init indexer and embedder
    logger.info("Loading embedding model...")
    embedder = Embedder()
    indexer = QdrantIndexer()

    db = get_db_session()

    stats = {
        "processed": 0,
        "skipped": 0,
        "failed": 0,
        "total_chunks": 0,
    }

    for i, file_path in enumerate(files, 1):
        print(f"\n[{i}/{len(files)}] Processing: {file_path.name}")
        print(f"            Path: {file_path}")

        # ── Step 1: Check if already processed ────────
        # use file path as unique URL
        file_url = f"file://{file_path.resolve()}"

        existing = db.query(Document).filter(
            Document.url == file_url
        ).first()

        if existing and existing.is_indexed:
            print(f"    ⏭️  Already indexed, skipping")
            stats["skipped"] += 1
            continue

        # ── Step 2: Detect metadata from path ─────────
        metadata = detect_metadata_from_path(file_path)
        print(
            f"    📁 Detected: "
            f"jurisdiction={metadata['jurisdiction']} | "
            f"topic={metadata['topic']}"
        )

        # ── Step 3: Extract text ───────────────────────
        extra_metadata = {}

        if file_path.suffix.lower() == ".pdf":
            text = extract_text_from_pdf(file_path)

        elif file_path.suffix.lower() == ".txt":
            text, extra_metadata = extract_text_from_txt(file_path)
            # override path metadata with file metadata if available
            if extra_metadata.get("jurisdiction"):
                metadata["jurisdiction"] = extra_metadata["jurisdiction"]
            if extra_metadata.get("topic"):
                metadata["topic"] = extra_metadata["topic"]
            if extra_metadata.get("law_name"):
                metadata["law_name"] = extra_metadata["law_name"]
            if extra_metadata.get("agency"):
                metadata["agency"] = extra_metadata["agency"]
        else:
            print(f"    ⚠️  Unsupported file type: {file_path.suffix}")
            stats["failed"] += 1
            continue

        if not text or len(text) < 100:
            print(f"    ❌ Could not extract text (got {len(text)} chars)")
            stats["failed"] += 1
            continue

        print(f"    📝 Extracted: {len(text):,} characters")

        # ── Step 4: Save to SQLite ─────────────────────
        try:
            source_url = extra_metadata.get("url", file_url)

            # check if URL already exists
            if extra_metadata.get("url"):
                existing_url = db.query(Document).filter(
                    Document.url == source_url
                ).first()
                if existing_url:
                    doc = existing_url
                else:
                    doc = None
            else:
                doc = existing

            if not doc:
                doc = Document(
                    id=str(uuid.uuid4()),
                    url=source_url,
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
                db.add(doc)
                db.commit()
                db.refresh(doc)
                print(f"    💾 Saved to database: {doc.id[:8]}...")
            else:
                # update existing
                doc.raw_text = text
                doc.is_indexed = False
                db.commit()
                print(f"    💾 Updated in database: {doc.id[:8]}...")

        except Exception as e:
            logger.error(f"Database save failed: {e}")
            db.rollback()
            stats["failed"] += 1
            continue

        # ── Step 5: Create chunks ──────────────────────
        chunks = create_chunks(
            text=text,
            document_metadata={
                "id": doc.id,
                "url": doc.url,
                "title": doc.title,
                "jurisdiction": doc.jurisdiction,
                "topic": doc.topic,
                "law_name": doc.law_name,
                "document_type": doc.document_type,
                "agency": doc.agency,
                "effective_date": doc.effective_date,
                "file_type": doc.file_type,
            }
        )

        if not chunks:
            print(f"    ❌ No chunks created")
            stats["failed"] += 1
            continue

        print(f"    🔪 Created {len(chunks)} chunks")

        # ── Step 6: Index into Qdrant ──────────────────
        try:
            chunk_count = indexer.index_chunks(
                chunks=chunks,
                embedder=embedder
            )

            # update document as indexed
            doc.is_indexed = True
            doc.chunk_count = chunk_count
            db.commit()

            # save index log
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
            print(f"    ✅ Indexed {chunk_count} chunks into Qdrant")

        except Exception as e:
            logger.error(f"Qdrant indexing failed: {e}")
            stats["failed"] += 1
            continue

    db.close()

    # ── Final Summary ──────────────────────────────────
    print("\n" + "=" * 60)
    print("INGESTION COMPLETE")
    print("=" * 60)
    print(f"✅ Processed:    {stats['processed']} documents")
    print(f"⏭️  Skipped:      {stats['skipped']} (already indexed)")
    print(f"❌ Failed:       {stats['failed']} documents")
    print(f"🔢 Total chunks: {stats['total_chunks']}")

    # show Qdrant stats
    try:
        total_vectors = indexer.count_points()
        print(f"📊 Qdrant total: {total_vectors} vectors")
    except Exception:
        pass

    print("=" * 60)
    print("\nNext step: python test_real_data.py")


if __name__ == "__main__":
    ingest_all_documents()