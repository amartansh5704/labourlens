# api/core/config.py
# Central config - reads all settings from .env file
# Import this anywhere you need settings

from dotenv import load_dotenv
from loguru import logger
import os

load_dotenv()


# api/core/config.py
# Add these new settings to the Settings class

class Settings:

    # existing settings...
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv(
        "GROQ_MODEL", "llama-3.3-70b-versatile"
    )
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION: str = os.getenv(
        "QDRANT_COLLECTION", "employment_law_india"
    )
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
    )
    EMBEDDING_DIMENSION: int = 384
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", "sqlite:///./laborlens.db"
    )

    # ── Updated chunk settings ─────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 400))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 80))

    # ── Retrieval settings ─────────────────────────────
    # TOP_K_RESULTS = final chunks sent to LLM
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 5))

    # RETRIEVAL_TOP_K = how many we fetch from Qdrant
    # should be 3-4x TOP_K_RESULTS for reranker to work
    RETRIEVAL_TOP_K: int = int(
        os.getenv("RETRIEVAL_TOP_K", 20)
    )

    # ── Reranker settings ──────────────────────────────
    RERANKER_MODEL: str = os.getenv(
        "RERANKER_MODEL",
        "cross-encoder/ms-marco-MiniLM-L-6-v2"
    )
    RERANKER_TOP_K: int = int(os.getenv("RERANKER_TOP_K", 5))
    ENABLE_RERANKER: bool = os.getenv(
        "ENABLE_RERANKER", "true"
    ).lower() == "true"

    # ── Score threshold ────────────────────────────────
    MIN_SCORE_THRESHOLD: float = float(
        os.getenv("MIN_SCORE_THRESHOLD", 0.15)
    )
    # lower threshold so we fetch more candidates
    # reranker will filter out irrelevant ones

    # ── API ────────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))
    MAX_QUESTION_LENGTH: int = 500
    MAX_ANSWER_TOKENS: int = int(
        os.getenv("MAX_ANSWER_TOKENS", 2048)
    )

    def validate(self):
        """Check all required settings are present"""
        errors = []

        if not self.GROQ_API_KEY:
            errors.append("GROQ_API_KEY is missing from .env")

        if not self.QDRANT_HOST:
            errors.append("QDRANT_HOST is missing from .env")

        if errors:
            for error in errors:
                logger.error(f"Config error: {error}")
            raise ValueError(
                f"Missing required settings: {errors}"
            )

        logger.info("Configuration validated successfully")
        return True


# single shared instance
# import this everywhere
settings = Settings()