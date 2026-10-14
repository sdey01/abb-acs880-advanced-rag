"""Lazy-loading CPU cross-encoder reranking."""
from __future__ import annotations

from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL


class CrossEncoderReranker:
    def __init__(self, model_name: str = RERANKER_MODEL) -> None:
        self.model_name, self._model = model_name, None

    def rerank(self, question: str, candidates):
        if not candidates:
            return []
        if self._model is None:
            self._model = CrossEncoder(self.model_name, device="cpu")
        scores = self._model.predict([(question, item.text) for item in candidates],
                                     show_progress_bar=False)
        ranked = sorted(zip(candidates, scores), key=lambda pair: float(pair[1]), reverse=True)
        for rank, (chunk, score) in enumerate(ranked, 1):
            chunk.rerank_score, chunk.reranked_rank = float(score), rank
        return [chunk for chunk, _ in ranked]
