# scraper/indexer/qdrant_indexer.py
# Stores embeddings and chunk data in Qdrant

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)
from scraper.indexer.embedder import Embedder
from typing import List, Dict
from loguru import logger
import uuid as uuid_lib
import os


class QdrantIndexer:
    """
    Manages storing vectors in Qdrant.
    Works with both local Docker and Qdrant Cloud.
    """

    def __init__(self):
        api_key = os.getenv("QDRANT_API_KEY", "")
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))

        # cloud Qdrant needs API key and uses HTTPS
        if api_key and "cloud.qdrant.io" in host:
            self.client = QdrantClient(
                url=f"https://{host}",
                api_key=api_key,
            )
            logger.info(f"Connected to Qdrant Cloud: {host}")
        else:
            # local Qdrant (no API key needed)
            self.client = QdrantClient(
                host=host,
                port=port
            )
            logger.info(f"Connected to local Qdrant: {host}:{port}")

        self.collection_name = os.getenv(
            "QDRANT_COLLECTION",
            "employment_law_india"
        )
        self.vector_size = 384
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        """Create collection if it does not exist yet"""
        try:
            existing = [
                c.name
                for c in self.client.get_collections().collections
            ]

            if self.collection_name not in existing:
                self.client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=VectorParams(
                        size=self.vector_size,
                        distance=Distance.COSINE
                    )
                )
                logger.info(
                    f"Created Qdrant collection: {self.collection_name}"
                )
            else:
                logger.info(
                    f"Collection exists: {self.collection_name}"
                )

        except Exception as e:
            logger.error(f"Failed to ensure collection: {e}")
            raise

    def index_chunks(
        self,
        chunks: List[Dict],
        embedder: Embedder
    ) -> int:
        """
        Embed chunks and store in Qdrant.

        Args:
            chunks:   list of chunk dicts from chunker.py
            embedder: Embedder instance

        Returns:
            number of chunks indexed
        """

        if not chunks:
            return 0

        texts = [chunk["text"] for chunk in chunks]
        embeddings = embedder.embed(texts)

        # build points with UUID string IDs
        points = []
        for chunk, embedding in zip(chunks, embeddings):
            point = PointStruct(
                id=str(uuid_lib.uuid4()),
                vector=embedding.tolist(),
                payload=chunk
            )
            points.append(point)

        # upsert in batches of 50
        batch_size = 50
        total_indexed = 0

        for batch_start in range(0, len(points), batch_size):
            batch = points[batch_start:batch_start + batch_size]

            self.client.upsert(
                collection_name=self.collection_name,
                points=batch,
                wait=True
            )
            total_indexed += len(batch)

        logger.info(
            f"Indexed {total_indexed} chunks "
            f"into {self.collection_name}"
        )

        return total_indexed

    def count_points(self) -> int:
        """Count total points in collection"""
        try:
            result = self.client.count(
                collection_name=self.collection_name,
                exact=True
            )
            return result.count
        except Exception as e:
            logger.warning(f"Could not count points: {e}")
            return 0

    def get_collection_stats(self) -> Dict:
        """Get stats about what is stored in Qdrant"""
        try:
            count = self.count_points()
            return {
                "total_vectors": count,
                "collection_name": self.collection_name,
                "vector_size": self.vector_size,
                "status": "ok"
            }
        except Exception as e:
            logger.warning(f"Could not get stats: {e}")
            return {
                "total_vectors": 0,
                "collection_name": self.collection_name,
                "vector_size": self.vector_size,
                "status": "error"
            }

    def delete_collection(self):
        """Delete and recreate collection for fresh start"""
        try:
            self.client.delete_collection(self.collection_name)
            logger.info(
                f"Deleted collection: {self.collection_name}"
            )
            self._ensure_collection_exists()
        except Exception as e:
            logger.error(f"Failed to delete collection: {e}")