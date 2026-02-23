from __future__ import annotations
import logging
from typing import List, Dict, Any, Optional

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranker using a local sentence-transformers cross-encoder model.

    Best accuracy among local options. Requires ~150 MB model download on first use.
    Recommended model: "cross-encoder/ms-marco-MiniLM-L-6-v2"
    Other options:
        - "cross-encoder/ms-marco-MiniLM-L-12-v2"  (slower, slightly better)
        - "cross-encoder/ms-marco-electra-base"     (larger, best quality)
    """

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        try:
            from sentence_transformers import CrossEncoder
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker.\n"
                "Install it with:  pip install sentence-transformers"
            )
        print(f"[CrossEncoderReranker] Loading model: {model_name}")
        self.model = CrossEncoder(model_name)
        print(f"[CrossEncoderReranker] Model loaded successfully")

    def rerank(
        self,
        query: str,
        docs: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        if not docs:
            return docs

        # Build (query, passage) pairs
        pairs = [(query, doc["text"]) for doc in docs]

        # Score all pairs at once (efficient batch inference)
        scores = self.model.predict(pairs)

        # Attach rerank_score to each doc
        for doc, score in zip(docs, scores):
            doc["rerank_score"] = float(score)

        # Sort by rerank_score descending
        reranked = sorted(docs, key=lambda d: d["rerank_score"], reverse=True)

        if top_k is not None:
            reranked = reranked[:top_k]

        return reranked

