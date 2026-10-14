"""Cache-first classifier, retrieval, reranking and grounded answer generation."""
from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import timedelta

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import (BROADER_CANDIDATES, FACTUAL_CANDIDATES, FINAL_CONTEXT_LIMIT,
                    OPENAI_MODEL)
from src.classifier import QuestionClassifier
from src.prompts import ANSWER_SYSTEM
from src.query_expansion import QueryExpander
from src.schemas import Answer, Chunk
from src.semantic_cache import SemanticCache, cacheable

LOGGER = logging.getLogger(__name__)


class RagPipeline:
    def __init__(self, index, cache: SemanticCache, reranker, llm=None) -> None:
        self.index, self.cache, self.reranker, self.llm = index, cache, reranker, llm

    def _llm(self):
        if self.llm is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise RuntimeError("OpenAI API key missing; add OPENAI_API_KEY to the existing .env file.")
            self.llm = ChatOpenAI(model=OPENAI_MODEL, temperature=None)
        return self.llm

    def ask(self, question: str) -> Answer:
        started, request_id = time.perf_counter(), uuid.uuid4().hex[:12]
        question = question.strip()
        if not question:
            raise ValueError("Enter a question before submitting.")
        timings, fallbacks = {}, []
        stage = time.perf_counter()
        try:
            match = self.cache.lookup(question)
        except Exception as exc:
            match = None
            fallbacks.append("Semantic cache unavailable; continuing without cache.")
            LOGGER.warning("request_id=%s cache_lookup_error=%s", request_id, type(exc).__name__)
        timings["cache_lookup"] = time.perf_counter() - stage
        if match:
            entry = match["entry"]
            result = Answer(
                question=question, answer=entry.answer, category=entry.category,
                cache_status="CACHE HIT", sources=[Chunk.from_dict(x) for x in entry.sources],
                reason="Returned from semantic cache.", latency_seconds=time.perf_counter()-started,
                timings=timings, request_id=request_id,
                cache_match_question=entry.question, cache_similarity=match["similarity"],
                cache_ttl_seconds=match["remaining_ttl"], cached_at=entry.created.isoformat(),
                expires_at=entry.expires.isoformat(),
            )
            self.cache.record({"status": "CACHE HIT", "question": question, "request_id": request_id})
            return result
        llm = self._llm()
        stage = time.perf_counter()
        classification, fallback = QuestionClassifier(llm).classify(question)
        timings["classification"] = time.perf_counter() - stage
        if fallback:
            fallbacks.append("Classifier unavailable or invalid; deterministic category fallback used.")
        stage = time.perf_counter()
        expanded, warning = QueryExpander(llm).expand(question)
        timings["expansion"] = time.perf_counter() - stage
        if warning:
            fallbacks.append(warning)
        queries = [question] + [q for q in expanded if q.casefold() != question.casefold()]
        budget = FACTUAL_CANDIDATES if classification.category == "FACTUAL" else BROADER_CANDIDATES
        stage = time.perf_counter()
        candidates = self.index.retrieve(queries, budget)
        timings["retrieval"] = time.perf_counter() - stage
        for rank, chunk in enumerate(candidates, 1):
            chunk.initial_rank = rank
        stage = time.perf_counter()
        try:
            ranked = self.reranker.rerank(question, candidates)
            if len(ranked) != len(candidates):
                raise ValueError("Reranker changed the candidate count.")
        except Exception as exc:
            LOGGER.warning("request_id=%s reranker_error=%s", request_id, type(exc).__name__)
            fallbacks.append("Reranking failed; original retrieval order retained.")
            ranked = candidates
            for rank, chunk in enumerate(ranked, 1):
                chunk.reranked_rank = rank
        timings["reranking"] = time.perf_counter() - stage
        selected = ranked[:FINAL_CONTEXT_LIMIT]
        stage = time.perf_counter()
        if selected:
            answer_text = self._generate(question, selected, llm)
        else:
            answer_text = ("The manual did not provide enough relevant text to answer this question. "
                           "Try a more specific query or check the indexed PDF.")
        timings["generation"] = time.perf_counter() - stage
        result = Answer(
            question=question, answer=answer_text, category=classification.category,
            cache_status="CACHE MISS", sources=selected, candidates=candidates,
            expanded_queries=expanded, reason=classification.reason,
            confidence=classification.confidence, timings=timings, fallbacks=fallbacks,
            request_id=request_id, candidate_budget=budget,
        )
        if cacheable(result):
            stage = time.perf_counter()
            try:
                expiry = self.cache.set(question, result)
                result.expires_at = expiry.isoformat()
                result.cached_at = (expiry - timedelta(seconds=self.cache.ttl)).isoformat()
            except Exception as exc:
                LOGGER.warning("request_id=%s cache_insert_error=%s", request_id, type(exc).__name__)
                result.fallbacks.append("Answer returned, but could not be stored in cache.")
            timings["cache_insert"] = time.perf_counter() - stage
        else:
            result.fallbacks.append("Answer was not cached because it lacked sufficient evidence.")
        result.latency_seconds = time.perf_counter() - started
        self.cache.record({"status": "CACHE MISS", "question": question, "request_id": request_id})
        LOGGER.info("request_id=%s category=%s cache=miss candidates=%d latency=%.3f",
                    request_id, result.category, len(candidates), result.latency_seconds)
        return result

    @staticmethod
    def _generate(question: str, sources: list[Chunk], llm) -> str:
        context = "\n\n".join(
            f"[{chunk.source}, p. {chunk.page if chunk.page is not None else 'page unavailable'}]\n{chunk.text}"
            for chunk in sources
        )
        response = llm.invoke([
            SystemMessage(content=ANSWER_SYSTEM),
            HumanMessage(content=f"Question:\n{question}\n\nManual excerpts:\n{context}"),
        ])
        text = response.content
        if not isinstance(text, str):
            text = "\n".join(part.get("text", "") for part in text if isinstance(part, dict))
        if not text.strip():
            raise RuntimeError("The answer model returned an empty response.")
        return text.strip()
