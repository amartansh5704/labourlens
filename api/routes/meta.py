# api/routes/meta.py
# Utility endpoints - health check, stats, jurisdictions etc

import os
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from database.connection import (
    get_db,
    check_db_connection,
    get_db_stats
)
from api.schemas.response import HealthResponse, StatsResponse
from api.core.config import settings
from shared.constants import JURISDICTIONS, TOPICS
from loguru import logger
from qdrant_client import QdrantClient

router = APIRouter()


def get_qdrant_client() -> QdrantClient:
    """
    Returns correct Qdrant client
    Works for both local Docker and Qdrant Cloud
    """
    api_key = os.getenv("QDRANT_API_KEY", "")
    host = os.getenv("QDRANT_HOST", "localhost")
    port = int(os.getenv("QDRANT_PORT", 6333))

    if api_key and "cloud.qdrant.io" in host:
        return QdrantClient(
            url=f"https://{host}",
            api_key=api_key,
        )
    else:
        return QdrantClient(
            host=host,
            port=port
        )


# ─────────────────────────────────────────────────────────
# GET /api/health
# Check all services are running
# ─────────────────────────────────────────────────────────
@router.get(
    "/health",
    response_model=HealthResponse,
    summary="Health check",
    description="Check if all services are running correctly."
)
def health_check():
    """
    Checks:
    - Database connection
    - Qdrant connection
    - Groq API connection
    """

    logger.info("GET /api/health")

    # check database
    db_status = "ok" if check_db_connection() else "error"

    # check Qdrant - handles both local and cloud
    try:
        client = get_qdrant_client()
        client.get_collections()
        qdrant_status = "ok"
    except Exception as e:
        logger.warning(f"Qdrant health check failed: {e}")
        qdrant_status = "error"

    # check Groq
    try:
        from groq import Groq
        groq_client = Groq(api_key=settings.GROQ_API_KEY)
        groq_client.chat.completions.create(
            model=settings.GROQ_MODEL,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5
        )
        groq_status = "ok"
    except Exception as e:
        logger.warning(f"Groq health check failed: {e}")
        groq_status = "error"

    # overall status
    overall = (
        "ok"
        if all(
            s == "ok"
            for s in [db_status, qdrant_status, groq_status]
        )
        else "degraded"
    )

    return HealthResponse(
        status=overall,
        qdrant=qdrant_status,
        database=db_status,
        groq=groq_status,
        version="1.0.0"
    )


# ─────────────────────────────────────────────────────────
# GET /api/jurisdictions
# List available jurisdictions
# ─────────────────────────────────────────────────────────
@router.get(
    "/jurisdictions",
    summary="List jurisdictions",
    description="Returns all available jurisdictions."
)
def get_jurisdictions():
    """Returns list of all supported jurisdictions"""
    return {
        "jurisdictions": list(JURISDICTIONS.keys()),
        "details": JURISDICTIONS
    }


# ─────────────────────────────────────────────────────────
# GET /api/topics
# List available topics
# ─────────────────────────────────────────────────────────
@router.get(
    "/topics",
    summary="List topics",
    description="Returns all available law topics."
)
def get_topics():
    """Returns list of all supported topics"""
    return {
        "topics": TOPICS,
        "keys": list(TOPICS.keys()),
        "names": list(TOPICS.values())
    }


# ─────────────────────────────────────────────────────────
# GET /api/stats
# Database and index statistics
# ─────────────────────────────────────────────────────────
@router.get(
    "/stats",
    response_model=StatsResponse,
    summary="System statistics",
    description="Returns counts of documents, chunks, and index stats."
)
def get_stats():
    """
    Returns:
    - Total documents scraped
    - Documents indexed into Qdrant
    - Chunks per jurisdiction
    - Qdrant vector count
    """

    logger.info("GET /api/stats")

    # get database stats
    db_stats = get_db_stats()

    # get Qdrant vector count
    # handles both local and cloud
    try:
        client = get_qdrant_client()
        qdrant_count = client.count(
            collection_name=settings.QDRANT_COLLECTION,
            exact=True
        ).count
    except Exception as e:
        logger.warning(f"Qdrant count failed: {e}")
        qdrant_count = 0

    return StatsResponse(
        total_documents=db_stats["total_documents"],
        indexed_documents=db_stats["indexed_documents"],
        unindexed_documents=db_stats["unindexed_documents"],
        total_chunks=db_stats["total_chunks"],
        total_scrape_attempts=db_stats["total_scrape_attempts"],
        failed_scrapes=db_stats["failed_scrapes"],
        by_jurisdiction=db_stats["by_jurisdiction"],
        by_topic=db_stats["by_topic"],
        qdrant_vectors=qdrant_count
    )