# scraper/indexer/embedder.py
# Converts text chunks into vector embeddings

from sentence_transformers import SentenceTransformer
from typing import List
import numpy as np
import os
from loguru import logger


class Embedder:
    """
    Converts text into vector embeddings using
    BAAI/bge-small-en-v1.5 model.

    This model:
    - Is only 130MB
    - Runs on CPU (no GPU needed)
    - Produces 384 dimensional vectors
    - Works well for legal/formal text
    """

    def __init__(self):
        model_name = os.getenv(
            "EMBEDDING_MODEL",
            "BAAI/bge-small-en-v1.5"
        )
        logger.info(f"Loading embedding model: {model_name}")

        self.model = SentenceTransformer(model_name)
        self.dimension = 384
        self.prefix = "Represent this Indian legal document: "

        logger.info("Embedding model loaded successfully")

    def embed(self, texts: List[str]) -> np.ndarray:
        """
        Convert list of texts to embeddings.

        Args:
            texts: list of text strings to embed

        Returns:
            numpy array of shape (len(texts), 384)
        """

        if not texts:
            return np.array([])

        # add prefix to help model understand domain
        prefixed_texts = [
            f"{self.prefix}{text}"
            for text in texts
        ]

        logger.debug(f"Embedding {len(texts)} texts...")

        embeddings = self.model.encode(
            prefixed_texts,
            batch_size=16,           # process 16 at a time
            show_progress_bar=False,
            normalize_embeddings=True # normalize for cosine similarity
        )

        logger.debug(
            f"Generated embeddings: shape={embeddings.shape}"
        )

        return embeddings

    def embed_query(self, query: str) -> List[float]:
        """
        Embed a single query text.
        Used at search time.

        Returns list of floats (not numpy array)
        """
        query_prefix = "Represent this legal query: "
        prefixed = f"{query_prefix}{query}"

        embedding = self.model.encode(
            prefixed,
            normalize_embeddings=True
        )

        return embedding.tolist()