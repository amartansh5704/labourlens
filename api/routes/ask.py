# api/routes/ask.py
# Handles the main question answering endpoints

from fastapi import APIRouter, HTTPException, Depends
from api.schemas.request import AskRequest, CompareRequest
from api.schemas.response import (
    AskResponse,
    CompareResponse,
    JurisdictionResult,
    SourceDocument
)
from api.rag.pipeline import LegalRAGPipeline
from loguru import logger

router = APIRouter()

# single pipeline instance shared across requests
# loads embedding model once at startup
_pipeline = None


def get_pipeline() -> LegalRAGPipeline:
    """
    Returns shared pipeline instance.
    Creates it on first call (lazy loading).
    """
    global _pipeline
    if _pipeline is None:
        logger.info("Creating RAG pipeline instance...")
        _pipeline = LegalRAGPipeline()
    return _pipeline


# ─────────────────────────────────────────────────────────
# POST /api/ask
# Main endpoint - answers a compliance question
# ─────────────────────────────────────────────────────────
@router.post(
    "/ask",
    response_model=AskResponse,
    summary="Ask a compliance question",
    description="Ask any Indian employment law compliance question. Optionally filter by jurisdiction and topic."
)
async def ask_question(
    request: AskRequest,
    pipeline: LegalRAGPipeline = Depends(get_pipeline)
):
    """
    Main RAG endpoint.

    - Searches indexed legal documents
    - Generates AI answer with citations
    - Returns sources with exact quotes
    """

    logger.info(
        f"POST /api/ask | "
        f"question='{request.question[:50]}' | "
        f"jurisdiction={request.jurisdiction} | "
        f"topic={request.topic}"
    )

    try:
        result = pipeline.run(
            question=request.question,
            jurisdiction=request.jurisdiction,
            topic=request.topic,
            top_k=request.top_k,
        )

        # convert source dicts to SourceDocument objects
        sources = [
            SourceDocument(**source)
            for source in result.get("sources", [])
        ]

        return AskResponse(
            answer=result["answer"],
            sources=sources,
            jurisdiction=result.get("jurisdiction"),
            topic=result.get("topic"),
            has_results=result.get("has_results", False),
            disclaimer=result["disclaimer"],
            top_score=result.get("top_score"),
            is_low_confidence=result.get("is_low_confidence", False),
        )

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Ask endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )


# ─────────────────────────────────────────────────────────
# POST /api/compare
# Compare same question across two jurisdictions
# ─────────────────────────────────────────────────────────
@router.post(
    "/compare",
    response_model=CompareResponse,
    summary="Compare two jurisdictions",
    description="Ask the same compliance question for two different jurisdictions and compare answers."
)
async def compare_jurisdictions(
    request: CompareRequest,
    pipeline: LegalRAGPipeline = Depends(get_pipeline)
):
    """
    Comparison endpoint.
    Runs the same question against two jurisdictions.
    """

    logger.info(
        f"POST /api/compare | "
        f"'{request.question[:50]}' | "
        f"{request.jurisdiction1} vs {request.jurisdiction2}"
    )

    if request.jurisdiction1 == request.jurisdiction2:
        raise HTTPException(
            status_code=400,
            detail="jurisdiction1 and jurisdiction2 must be different"
        )

    try:
        result = pipeline.run_comparison(
            question=request.question,
            jurisdiction1=request.jurisdiction1,
            jurisdiction2=request.jurisdiction2,
            topic=request.topic,
        )

        # build comparison dict with proper types
        comparison = {}
        for jurisdiction, data in result["comparison"].items():
            sources = [
                SourceDocument(**s)
                for s in data.get("sources", [])
            ]
            comparison[jurisdiction] = JurisdictionResult(
                answer=data["answer"],
                sources=sources,
                has_results=data["has_results"],
            )

        return CompareResponse(
            question=result["question"],
            topic=result.get("topic"),
            comparison=comparison,
            disclaimer=result["disclaimer"],
        )

    except HTTPException:
        raise

    except Exception as e:
        logger.error(f"Compare endpoint error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal error: {str(e)}"
        )