# api/core/config.py
# Central config - reads all settings from .env file
# Import this anywhere you need settings

from dotenv import load_dotenv
from loguru import logger
import os

load_dotenv()


class Settings:
    """
    All app settings loaded from .env file.
    Access anywhere with:
        from api.core.config import settings
        print(settings.GROQ_API_KEY)
    """

    # ── Groq LLM ──────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY", "")
    GROQ_MODEL: str = os.getenv(
        "GROQ_MODEL",
        "llama-3.3-70b-versatile"
    )

    # ── Qdrant ────────────────────────────────────────
    QDRANT_HOST: str = os.getenv("QDRANT_HOST", "localhost")
    QDRANT_PORT: int = int(os.getenv("QDRANT_PORT", 6333))
    QDRANT_COLLECTION: str = os.getenv(
        "QDRANT_COLLECTION",
        "employment_law_india"
    )

    # ── Embedding Model ───────────────────────────────
    EMBEDDING_MODEL: str = os.getenv(
        "EMBEDDING_MODEL",
        "BAAI/bge-small-en-v1.5"
    )
    EMBEDDING_DIMENSION: int = 384

    # ── Database ──────────────────────────────────────
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "sqlite:///./laborlens.db"
    )

    # ── RAG Settings ──────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 512))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 100))
    TOP_K_RESULTS: int = int(os.getenv("TOP_K_RESULTS", 5))

    # minimum similarity score to consider a result relevant
    # 0.0 = accept everything
    # 1.0 = only exact matches
    # 0.25 is a good balance for legal text
    MIN_SCORE_THRESHOLD: float = 0.20

    # ── API ───────────────────────────────────────────
    API_HOST: str = os.getenv("API_HOST", "0.0.0.0")
    API_PORT: int = int(os.getenv("API_PORT", 8000))

    # ── Validation ────────────────────────────────────
    MAX_QUESTION_LENGTH: int = 500
    MAX_ANSWER_TOKENS: int = 2048

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