# scraper/processors/chunker.py
# Splits large documents into smaller chunks
# Each chunk is what gets embedded and stored in Qdrant

from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List, Dict
from loguru import logger
import uuid
import os


def create_chunks(
    text: str,
    document_metadata: Dict,
) -> List[Dict]:
    """
    Split document text into chunks for embedding.

    Args:
        text: cleaned document text
        document_metadata: dict with document info
                          (id, url, jurisdiction, topic etc)

    Returns:
        List of chunk dicts ready to be embedded and indexed
    """

    if not text or len(text.strip()) < 100:
        logger.warning(
            f"Text too short to chunk: "
            f"{document_metadata.get('url', 'unknown')}"
        )
        return []

    chunk_size = int(os.getenv("CHUNK_SIZE", 512))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", 100))

    # Create splitter with legal document aware separators
    # It tries to split on these in order
    # If chunk still too big it moves to next separator
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=[
            "\n\n",        # paragraph breaks (preferred)
            "\nSection ",  # section headers
            "\nRule ",     # rule headers
            "\nClause ",   # clause headers
            "\nArticle ",  # article headers
            "\nSchedule ", # schedule headers
            "\n§",         # section symbols
            "\n(",         # numbered items
            "\n-",         # bullet points
            "\n",          # any line break
            ". ",          # sentence boundaries
            " ",           # words (last resort)
        ],
        length_function=len,
    )

    # split the text
    text_chunks = splitter.split_text(text)

    logger.debug(
        f"Split into {len(text_chunks)} chunks "
        f"from {len(text)} chars"
    )

    # build chunk objects with metadata
    chunks = []

    for i, chunk_text in enumerate(text_chunks):
        chunk_text = chunk_text.strip()

        # skip chunks that are too small to be useful
        if len(chunk_text) < 50:
            continue

        chunk = {
            # unique ID for this chunk
            "chunk_id": str(uuid.uuid4()),

            # link back to parent document
            "document_id": document_metadata.get("id", ""),

            # the actual text content
            "text": chunk_text,

            # position in document
            "chunk_index": i,
            "total_chunks": len(text_chunks),

            # metadata copied from document
            # these are used for filtering in Qdrant
            "jurisdiction": document_metadata.get("jurisdiction", ""),
            "topic": document_metadata.get("topic", ""),
            "law_name": document_metadata.get("law_name", ""),
            "document_type": document_metadata.get("document_type", ""),
            "agency": document_metadata.get("agency", ""),
            "source_url": document_metadata.get("url", ""),
            "title": document_metadata.get("title", ""),
            "effective_date": document_metadata.get("effective_date", ""),
            "file_type": document_metadata.get("file_type", "html"),
        }

        chunks.append(chunk)

    logger.info(
        f"Created {len(chunks)} valid chunks "
        f"for: {document_metadata.get('url', 'unknown')[:60]}"
    )

    return chunks