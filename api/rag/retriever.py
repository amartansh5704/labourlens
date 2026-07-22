# api/rag/retriever.py
# Searches Qdrant vector database for relevant law passages

import os
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Filter,
    FieldCondition,
    MatchValue,
)
from sentence_transformers import SentenceTransformer
from api.core.config import settings
from loguru import logger
from typing import List, Dict, Optional


class LegalRetriever:
    """
    Retrieves relevant legal document chunks from Qdrant.

    How it works:
    1. Convert question to embedding vector
    2. Filter by jurisdiction and/or topic if specified
    3. Find top K most similar vectors in Qdrant
    4. Return the chunk text and metadata
    """

    def __init__(self):
        api_key = os.getenv("QDRANT_API_KEY", "")
        host = os.getenv("QDRANT_HOST", "localhost")
        port = int(os.getenv("QDRANT_PORT", 6333))

        # cloud connection
        if api_key and "cloud.qdrant.io" in host:
            self.client = QdrantClient(
                url=f"https://{host}",
                api_key=api_key,
            )
            logger.info(
                f"Connected to Qdrant Cloud: {host}"
            )
        else:
            # local connection
            self.client = QdrantClient(
                host=host,
                port=port
            )
            logger.info(
                f"Connected to local Qdrant: {host}:{port}"
            )

        self.collection_name = os.getenv(
            "QDRANT_COLLECTION",
            "employment_law_india"
        )

        logger.info(
            f"Loading retriever model: {settings.EMBEDDING_MODEL}"
        )
        self.model = SentenceTransformer(settings.EMBEDDING_MODEL)
        logger.info("Retriever ready")

    def retrieve(
        self,
        question: str,
        jurisdiction: Optional[str] = None,
        topic: Optional[str] = None,
        top_k: int = None,
    ) -> List[Dict]:
        """
        Main retrieval function.

        Args:
            question:     user question in plain English
            jurisdiction: filter eg "Delhi"
            topic:        filter eg "minimum_wage"
            top_k:        how many results to return

        Returns:
            List of dicts with text and metadata
            Sorted by relevance score highest first
        """

        top_k = top_k or settings.TOP_K_RESULTS

        logger.info(
            f"Retrieving: '{question[:60]}' "
            f"| jurisdiction={jurisdiction} "
            f"| topic={topic} "
            f"| top_k={top_k}"
        )

        # step 1: embed the question
        query_vector = self._embed_question(question)

        # step 2: build metadata filter
        search_filter = self._build_filter(jurisdiction, topic)

        # step 3: search Qdrant
        try:
            results = self.client.query_points(
                collection_name=self.collection_name,
                query=query_vector,
                query_filter=search_filter,
                limit=top_k,
                with_payload=True,
                score_threshold=settings.MIN_SCORE_THRESHOLD,
            )

            raw_points = results.points

        except Exception as e:
            logger.error(f"Qdrant search failed: {e}")
            return []

        # step 4: format results
        formatted = self._format_results(raw_points)

        logger.info(
            f"Retrieved {len(formatted)} results "
            f"(threshold={settings.MIN_SCORE_THRESHOLD})"
        )

        return formatted

    def retrieve_for_comparison(
        self,
        question: str,
        jurisdiction1: str,
        jurisdiction2: str,
        top_k: int = 3,
    ) -> Dict:
        """
        Retrieve results for two jurisdictions separately.
        """
        results1 = self.retrieve(
            question=question,
            jurisdiction=jurisdiction1,
            top_k=top_k
        )
        results2 = self.retrieve(
            question=question,
            jurisdiction=jurisdiction2,
            top_k=top_k
        )

        return {
            jurisdiction1: results1,
            jurisdiction2: results2,
        }

    def _embed_question(self, question: str) -> List[float]:
        """Convert question text to embedding vector"""
        prefixed = f"Represent this legal query: {question}"

        embedding = self.model.encode(
            prefixed,
            normalize_embeddings=True
        )

        return embedding.tolist()

    def _build_filter(
        self,
        jurisdiction: Optional[str],
        topic: Optional[str]
    ) -> Optional[Filter]:
        """
        Build Qdrant filter from jurisdiction and topic.
        Returns None if no filters needed.
        """

        conditions = []

        if jurisdiction and jurisdiction.lower() != "all":
            conditions.append(
                FieldCondition(
                    key="jurisdiction",
                    match=MatchValue(value=jurisdiction)
                )
            )

        if topic and topic.lower() != "all":
            conditions.append(
                FieldCondition(
                    key="topic",
                    match=MatchValue(value=topic)
                )
            )

        if conditions:
            return Filter(must=conditions)

        return None

    def _format_results(self, raw_points: list) -> List[Dict]:
        """
        Convert raw Qdrant points to clean dicts.
        """
        formatted = []

        for point in raw_points:
            payload = point.payload or {}

            formatted.append({
                "text": payload.get("text", ""),
                "score": round(float(point.score), 4),
                "jurisdiction": payload.get(
                    "jurisdiction", ""
                ),
                "topic": payload.get("topic", ""),
                "law_name": payload.get("law_name", ""),
                "document_type": payload.get(
                    "document_type", ""
                ),
                "agency": payload.get("agency", ""),
                "source_url": payload.get("source_url", ""),
                "title": payload.get("title", ""),
                "effective_date": payload.get(
                    "effective_date", ""
                ),
                "file_type": payload.get("file_type", "html"),
                "chunk_index": payload.get("chunk_index", 0),
            })

        return formatted

    def count_indexed_documents(
        self,
        jurisdiction: Optional[str] = None,
        topic: Optional[str] = None,
    ) -> int:
        """Count how many chunks are in Qdrant"""
        try:
            search_filter = self._build_filter(
                jurisdiction, topic
            )

            result = self.client.count(
                collection_name=self.collection_name,
                count_filter=search_filter,
                exact=True,
            )
            return result.count

        except Exception as e:
            logger.error(f"Count failed: {e}")
            return 0