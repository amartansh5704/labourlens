# api/routes/documents.py
# Endpoints for viewing scraped documents

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session
from database.connection import get_db
from database.models import Document
from api.schemas.response import DocumentsResponse, DocumentInfo
from typing import Optional
from loguru import logger

router = APIRouter()


# ─────────────────────────────────────────────────────────
# GET /api/documents
# List all scraped documents with pagination
# ─────────────────────────────────────────────────────────
@router.get(
    "/documents",
    response_model=DocumentsResponse,
    summary="List all documents",
    description="Browse all scraped and indexed legal documents."
)
def get_documents(
    jurisdiction: Optional[str] = Query(
        None,
        description="Filter by jurisdiction eg Delhi"
    ),
    topic: Optional[str] = Query(
        None,
        description="Filter by topic eg minimum_wage"
    ),
    is_indexed: Optional[bool] = Query(
        None,
        description="Filter by indexed status"
    ),
    page: int = Query(1, ge=1, description="Page number"),
    page_size: int = Query(
        20, ge=1, le=100,
        description="Items per page"
    ),
    db: Session = Depends(get_db)
):
    """
    Returns paginated list of all scraped documents.
    Supports filtering by jurisdiction, topic, indexed status.
    """

    logger.info(
        f"GET /api/documents | "
        f"jurisdiction={jurisdiction} | "
        f"topic={topic} | "
        f"page={page}"
    )

    # build query with filters
    query = db.query(Document)

    if jurisdiction and jurisdiction != "All":
        query = query.filter(Document.jurisdiction == jurisdiction)

    if topic and topic != "All":
        query = query.filter(Document.topic == topic)

    if is_indexed is not None:
        query = query.filter(Document.is_indexed == is_indexed)

    # get total count
    total = query.count()

    # apply pagination
    offset = (page - 1) * page_size
    documents = query.offset(offset).limit(page_size).all()

    # convert to response model
    doc_list = [
        DocumentInfo(**doc.to_dict())
        for doc in documents
    ]

    return DocumentsResponse(
        documents=doc_list,
        total=total,
        page=page,
        page_size=page_size
    )


# ─────────────────────────────────────────────────────────
# GET /api/documents/{doc_id}
# Get a single document by ID
# ─────────────────────────────────────────────────────────
@router.get(
    "/documents/{doc_id}",
    response_model=DocumentInfo,
    summary="Get document by ID"
)
def get_document(
    doc_id: str,
    db: Session = Depends(get_db)
):
    """Returns a single document by its ID"""

    doc = db.query(Document).filter(
        Document.id == doc_id
    ).first()

    if not doc:
        from fastapi import HTTPException
        raise HTTPException(
            status_code=404,
            detail=f"Document {doc_id} not found"
        )

    return DocumentInfo(**doc.to_dict())