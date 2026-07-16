# database/models.py
# Defines what our database tables look like
# Think of each class as one table

from sqlalchemy import (
    Column,
    String,
    Text,
    Boolean,
    Integer,
    DateTime,
    Float
)
from sqlalchemy.orm import declarative_base
from datetime import datetime
import uuid

# Base class that all models inherit from
Base = declarative_base()


def generate_uuid():
    """Generate a unique ID for each record"""
    return str(uuid.uuid4())


# ─────────────────────────────────────────────────────────
# TABLE 1: Documents
# Stores every web page or PDF we scrape
# ─────────────────────────────────────────────────────────
class Document(Base):
    __tablename__ = "documents"

    # unique identifier for each document
    id = Column(
        String(36),
        primary_key=True,
        default=generate_uuid
    )

    # where we got it from
    url = Column(
        String(2048),
        unique=True,
        nullable=False,
        index=True          # index for faster lookup
    )

    # basic info
    title = Column(String(512))
    raw_text = Column(Text)           # full extracted text

    # classification
    jurisdiction = Column(
        String(100),
        nullable=False,
        index=True
    )
    topic = Column(
        String(100),
        nullable=False,
        index=True
    )
    document_type = Column(String(50))  # statute, notification etc
    law_name = Column(String(512))
    agency = Column(String(256))        # which govt dept
    effective_date = Column(String(50)) # when law came into effect

    # tracking
    scraped_at = Column(DateTime, default=datetime.utcnow)
    is_indexed = Column(Boolean, default=False)  # sent to Qdrant?
    chunk_count = Column(Integer, default=0)      # how many chunks made
    file_type = Column(String(10), default="html") # html or pdf

    def __repr__(self):
        return (
            f"<Document("
            f"id={self.id[:8]}..., "
            f"jurisdiction={self.jurisdiction}, "
            f"topic={self.topic}, "
            f"url={self.url[:50]}..."
            f")>"
        )

    def to_dict(self):
        """Convert to dictionary for API responses"""
        return {
            "id": self.id,
            "url": self.url,
            "title": self.title,
            "jurisdiction": self.jurisdiction,
            "topic": self.topic,
            "document_type": self.document_type,
            "law_name": self.law_name,
            "agency": self.agency,
            "effective_date": self.effective_date,
            "scraped_at": str(self.scraped_at),
            "is_indexed": self.is_indexed,
            "chunk_count": self.chunk_count,
            "file_type": self.file_type,
            "text_preview": (
                self.raw_text[:200] + "..."
                if self.raw_text and len(self.raw_text) > 200
                else self.raw_text
            )
        }


# ─────────────────────────────────────────────────────────
# TABLE 2: ScrapeLog
# Records every scraping attempt (success or failure)
# ─────────────────────────────────────────────────────────
class ScrapeLog(Base):
    __tablename__ = "scrape_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    # what we tried to scrape
    url = Column(String(2048))
    spider_name = Column(String(100))
    jurisdiction = Column(String(100))

    # what happened
    status = Column(String(20))    # success / failed / skipped
    error_msg = Column(Text)       # error details if failed
    http_status = Column(Integer)  # 200, 404, 500 etc

    # results
    scraped_at = Column(DateTime, default=datetime.utcnow)
    chunks_created = Column(Integer, default=0)
    text_length = Column(Integer, default=0)  # chars extracted

    def __repr__(self):
        return (
            f"<ScrapeLog("
            f"url={self.url[:40]}..., "
            f"status={self.status}"
            f")>"
        )


# ─────────────────────────────────────────────────────────
# TABLE 3: IndexLog
# Records every time we index chunks into Qdrant
# ─────────────────────────────────────────────────────────
class IndexLog(Base):
    __tablename__ = "index_logs"

    id = Column(Integer, primary_key=True, autoincrement=True)

    document_id = Column(String(36))
    document_url = Column(String(2048))
    status = Column(String(20))      # success / failed
    chunks_indexed = Column(Integer, default=0)
    error_msg = Column(Text)
    indexed_at = Column(DateTime, default=datetime.utcnow)

    def __repr__(self):
        return (
            f"<IndexLog("
            f"doc={self.document_id[:8]}..., "
            f"chunks={self.chunks_indexed}, "
            f"status={self.status}"
            f")>"
        )