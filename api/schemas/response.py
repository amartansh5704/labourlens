# api/schemas/response.py
# Defines what the API returns in responses
# Pydantic serializes all outgoing data automatically

from pydantic import BaseModel
from typing import Optional, List, Dict, Any


class SourceDocument(BaseModel):
    """
    One source document returned with an answer.
    Shows exactly which law passage was used.
    """
    text: str
    score: float
    jurisdiction: str
    topic: str
    law_name: str
    document_type: Optional[str] = ""
    agency: Optional[str] = ""
    source_url: Optional[str] = ""
    title: Optional[str] = ""
    effective_date: Optional[str] = ""
    file_type: Optional[str] = "html"
    chunk_index: Optional[int] = 0


class AskResponse(BaseModel):
    """
    Response for POST /api/ask
    Contains the answer and all source documents
    """
    answer: str
    sources: List[SourceDocument]
    jurisdiction: Optional[str]
    topic: Optional[str]
    has_results: bool
    disclaimer: str
    top_score: Optional[float] = None
    is_low_confidence: Optional[bool] = False

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "The minimum wage for unskilled workers in Delhi is Rs. 17,494 per month...",
                "sources": [
                    {
                        "text": "The minimum wages for unskilled workers in Delhi shall be Rs. 17,494...",
                        "score": 0.87,
                        "jurisdiction": "Delhi",
                        "topic": "minimum_wage",
                        "law_name": "Delhi Minimum Wages Notification 2023",
                        "agency": "Delhi Labour Department",
                        "source_url": "https://labour.delhi.gov.in/minimum-wages",
                        "effective_date": "October 2023"
                    }
                ],
                "jurisdiction": "Delhi",
                "topic": "minimum_wage",
                "has_results": True,
                "disclaimer": "This is for informational purposes only."
            }
        }


class JurisdictionResult(BaseModel):
    """Result for one jurisdiction in a comparison"""
    answer: str
    sources: List[SourceDocument]
    has_results: bool


class CompareResponse(BaseModel):
    """
    Response for POST /api/compare
    Contains answers for both jurisdictions side by side
    """
    question: str
    topic: Optional[str]
    comparison: Dict[str, JurisdictionResult]
    disclaimer: str


class DocumentInfo(BaseModel):
    """Info about one scraped document"""
    id: str
    url: str
    title: Optional[str]
    jurisdiction: str
    topic: str
    document_type: Optional[str]
    law_name: Optional[str]
    agency: Optional[str]
    effective_date: Optional[str]
    scraped_at: Optional[str]
    is_indexed: bool
    chunk_count: int
    file_type: Optional[str]
    text_preview: Optional[str]


class DocumentsResponse(BaseModel):
    """Response for GET /api/documents"""
    documents: List[DocumentInfo]
    total: int
    page: int
    page_size: int


class StatsResponse(BaseModel):
    """Response for GET /api/stats"""
    total_documents: int
    indexed_documents: int
    unindexed_documents: int
    total_chunks: int
    total_scrape_attempts: int
    failed_scrapes: int
    by_jurisdiction: Dict[str, int]
    by_topic: Dict[str, int]
    qdrant_vectors: int


class HealthResponse(BaseModel):
    """Response for GET /api/health"""
    status: str
    qdrant: str
    database: str
    groq: str
    version: str = "1.0.0"


class ErrorResponse(BaseModel):
    """Standard error response"""
    error: str
    detail: Optional[str] = None
    status_code: int