# database/connection.py
# Handles connecting to SQLite database
# and creating tables if they don't exist

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker, Session
from database.models import Base
from loguru import logger
import os
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────────────────
# DATABASE SETUP
# ─────────────────────────────────────────────────────────
DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./laborlens.db"
)

engine = create_engine(
    DATABASE_URL,
    connect_args={
        "check_same_thread": False  # needed for SQLite with FastAPI
    },
    echo=False  # set True to see all SQL queries in terminal
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine
)


# ─────────────────────────────────────────────────────────
# INIT FUNCTION
# Creates all tables if they don't exist
# Safe to call multiple times
# ─────────────────────────────────────────────────────────
def init_db():
    """
    Create all database tables.
    Call this once when app starts.
    If tables already exist, does nothing.
    """
    try:
        Base.metadata.create_all(bind=engine)
        logger.info("Database initialized successfully")
        logger.info(f"Database location: {DATABASE_URL}")
        return True
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")
        raise


# ─────────────────────────────────────────────────────────
# SESSION HELPERS
# ─────────────────────────────────────────────────────────
def get_db():
    """
    Provides a database session.
    Used as a dependency in FastAPI routes.

    Usage in FastAPI:
        @app.get("/something")
        def route(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_db_session() -> Session:
    """
    Returns a database session directly.
    Used in scripts (not FastAPI routes).

    Usage in scripts:
        db = get_db_session()
        docs = db.query(Document).all()
        db.close()
    """
    return SessionLocal()


# ─────────────────────────────────────────────────────────
# HEALTH CHECK
# ─────────────────────────────────────────────────────────
def check_db_connection() -> bool:
    """
    Test if database is reachable.
    Returns True if ok, False if not.
    """
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        return True
    except Exception as e:
        logger.error(f"Database connection check failed: {e}")
        return False


# ─────────────────────────────────────────────────────────
# STATS HELPER
# ─────────────────────────────────────────────────────────
def get_db_stats() -> dict:
    """
    Returns counts of records in each table.
    Useful for dashboard and API stats endpoint.
    """
    from database.models import Document, ScrapeLog, IndexLog

    db = get_db_session()
    try:
        total_docs = db.query(Document).count()
        indexed_docs = db.query(Document).filter(
            Document.is_indexed == True
        ).count()
        total_scrape_logs = db.query(ScrapeLog).count()
        failed_scrapes = db.query(ScrapeLog).filter(
            ScrapeLog.status == "failed"
        ).count()

        # count by jurisdiction
        by_jurisdiction = {}
        from shared.constants import JURISDICTION_NAMES
        for j in JURISDICTION_NAMES:
            count = db.query(Document).filter(
                Document.jurisdiction == j
            ).count()
            by_jurisdiction[j] = count

        # count by topic
        by_topic = {}
        from shared.constants import TOPIC_KEYS
        for t in TOPIC_KEYS:
            count = db.query(Document).filter(
                Document.topic == t
            ).count()
            by_topic[t] = count

        # total chunks across all docs
        from sqlalchemy import func
        total_chunks = db.query(
            func.sum(Document.chunk_count)
        ).scalar() or 0

        return {
            "total_documents": total_docs,
            "indexed_documents": indexed_docs,
            "unindexed_documents": total_docs - indexed_docs,
            "total_chunks": total_chunks,
            "total_scrape_attempts": total_scrape_logs,
            "failed_scrapes": failed_scrapes,
            "by_jurisdiction": by_jurisdiction,
            "by_topic": by_topic
        }

    finally:
        db.close()