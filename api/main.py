# api/main.py
# FastAPI application entry point
# This is what uvicorn runs

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api.routes import ask, documents, meta
from database.connection import init_db
from api.core.config import settings
from loguru import logger
import time

# ─────────────────────────────────────────────────────────
# CREATE FASTAPI APP
# ─────────────────────────────────────────────────────────
app = FastAPI(
    title="LaborLens API",
    description="""
    ## Indian Employment Law Compliance RAG System

    Ask compliance questions about Indian labor laws
    across multiple jurisdictions.

    ### Features
    - Semantic search across scraped legal documents
    - Filter by jurisdiction and topic
    - AI-generated answers with exact citations
    - Compare laws across two jurisdictions

    ### Jurisdictions Covered
    Central, Delhi, Maharashtra, Karnataka, Tamil Nadu, Telangana

    ### Topics Covered
    Minimum Wage, Working Hours, Leave Policy, EPF/ESI,
    Worker Classification
    """,
    version="1.0.0",
    docs_url="/docs",       # Swagger UI at /docs
    redoc_url="/redoc",     # ReDoc UI at /redoc
)

# ─────────────────────────────────────────────────────────
# MIDDLEWARE
# ─────────────────────────────────────────────────────────

# CORS - allows frontend to call the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # in production restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# request logging middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log every request with timing"""
    start = time.time()
    response = await call_next(request)
    duration = round((time.time() - start) * 1000, 2)

    logger.info(
        f"{request.method} {request.url.path} "
        f"| status={response.status_code} "
        f"| {duration}ms"
    )
    return response


# ─────────────────────────────────────────────────────────
# STARTUP EVENT
# ─────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    """Runs when API starts up"""
    logger.info("=" * 50)
    logger.info("LaborLens API Starting...")
    logger.info("=" * 50)

    # initialize database tables
    init_db()

    # validate config
    settings.validate()

    logger.info(f"Groq model:    {settings.GROQ_MODEL}")
    logger.info(f"Qdrant:        {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    logger.info(f"Collection:    {settings.QDRANT_COLLECTION}")
    logger.info(f"Docs at:       http://localhost:{settings.API_PORT}/docs")
    logger.info("=" * 50)
    logger.info("LaborLens API Ready")
    logger.info("=" * 50)


# ─────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────
app.include_router(
    ask.router,
    prefix="/api",
    tags=["Questions"]
)
app.include_router(
    documents.router,
    prefix="/api",
    tags=["Documents"]
)
app.include_router(
    meta.router,
    prefix="/api",
    tags=["System"]
)


# ─────────────────────────────────────────────────────────
# ROOT ENDPOINT
# ─────────────────────────────────────────────────────────
@app.get("/", tags=["Root"])
async def root():
    """Root endpoint - basic info"""
    return {
        "name": "LaborLens",
        "version": "1.0.0",
        "description": "Indian Employment Law Compliance RAG",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "ask": "POST /api/ask",
            "compare": "POST /api/compare",
            "documents": "GET /api/documents",
            "jurisdictions": "GET /api/jurisdictions",
            "topics": "GET /api/topics",
            "stats": "GET /api/stats",
            "health": "GET /api/health",
        }
    }


# ─────────────────────────────────────────────────────────
# GLOBAL ERROR HANDLER
# ─────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch any unhandled exceptions"""
    logger.error(f"Unhandled error: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc)
        }
    )