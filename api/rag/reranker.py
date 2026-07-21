# api/rag/reranker.py
# Reranks retrieved chunks by true relevance
# Uses cross-encoder model - more accurate than bi-encoder
# Cross-encoder looks at query AND document together
# Bi-encoder (what Qdrant uses) looks at them separately

from sentence_transformers import CrossEncoder
from loguru import logger
from typing import List, Dict, Optional
import os


class LegalReranker:
    """
    Reranks retrieved chunks using a Cross-Encoder model.

    How it works:
    ─────────────────────────────────────────────────────
    Bi-encoder (Qdrant retrieval):
      Encodes question separately: [0.2, 0.8, ...]
      Encodes document separately: [0.21, 0.79, ...]
      Compares vectors (fast but less accurate)

    Cross-encoder (Reranker):
      Looks at question + document TOGETHER
      "Does this specific document answer this question?"
      Gives a relevance score 0.0 to 1.0
      Much more accurate but slower
      Only used on top candidates, not all 3000 vectors

    Flow:
      Qdrant retrieves top 20 candidates (fast)
      Reranker scores all 20 against question (accurate)
      Returns top 5 most relevant (best quality)
    """

    # Models from best to smallest/fastest
    # cross-encoder/ms-marco-MiniLM-L-6-v2 = 80MB, very fast
    # cross-encoder/ms-marco-MiniLM-L-12-v2 = 130MB, better
    # cross-encoder/ms-marco-electra-base = 400MB, best

    DEFAULT_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    def __init__(self):
        model_name = os.getenv(
            "RERANKER_MODEL",
            self.DEFAULT_MODEL
        )

        logger.info(f"Loading reranker model: {model_name}")

        try:
            self.model = CrossEncoder(
                model_name,
                max_length=512,
                device="cpu"  # always CPU, no GPU needed
            )
            self.enabled = True
            logger.info("Reranker loaded successfully")

        except Exception as e:
            logger.warning(
                f"Reranker failed to load: {e}. "
                f"Falling back to Qdrant scores."
            )
            self.model = None
            self.enabled = False

    def rerank(
        self,
        question: str,
        chunks: List[Dict],
        top_k: int = 5,
    ) -> List[Dict]:
        """
        Rerank chunks by true relevance to question.

        Args:
            question: user's original question
            chunks:   list of chunk dicts from Qdrant
            top_k:    how many to return after reranking

        Returns:
            top_k most relevant chunks sorted by rerank score
        """

        if not chunks:
            return []

        # if reranker not loaded, fall back to Qdrant scores
        if not self.enabled or self.model is None:
            logger.warning(
                "Reranker disabled, using Qdrant scores"
            )
            return chunks[:top_k]

        # if only a few chunks, no need to rerank
        if len(chunks) <= top_k:
            logger.debug(
                f"Only {len(chunks)} chunks, skipping rerank"
            )
            return chunks

        logger.info(
            f"Reranking {len(chunks)} chunks → top {top_k}"
        )

        try:
            # build pairs: [question, chunk_text] for each chunk
            pairs = [
                [question, chunk["text"]]
                for chunk in chunks
            ]

            # cross-encoder scores each pair
            # returns float score for each pair
            scores = self.model.predict(
                pairs,
                show_progress_bar=False
            )

            # attach rerank scores to chunks
            for i, chunk in enumerate(chunks):
                chunk["rerank_score"] = float(scores[i])
                # keep original Qdrant score too
                chunk["qdrant_score"] = chunk.get("score", 0)

            # sort by rerank score (highest first)
            reranked = sorted(
                chunks,
                key=lambda x: x["rerank_score"],
                reverse=True
            )

            # take top_k
            final = reranked[:top_k]

            logger.info(
                f"Reranking complete. "
                f"Top score: {final[0]['rerank_score']:.4f} | "
                f"Bottom score: {final[-1]['rerank_score']:.4f}"
            )

            # log what changed position
            original_order = [c.get("law_name", "")[:30] for c in chunks[:top_k]]
            new_order = [c.get("law_name", "")[:30] for c in final]
            if original_order != new_order:
                logger.debug(f"Order changed after reranking")
                logger.debug(f"Before: {original_order}")
                logger.debug(f"After:  {new_order}")

            return final

        except Exception as e:
            logger.error(f"Reranking failed: {e}")
            # safe fallback
            return chunks[:top_k]

    def score_single(
        self,
        question: str,
        text: str,
    ) -> float:
        """
        Score a single question-document pair.
        Useful for testing.
        """
        if not self.enabled:
            return 0.0

        try:
            score = self.model.predict([[question, text]])
            return float(score[0])
        except Exception as e:
            logger.error(f"Single score failed: {e}")
            return 0.0