"""Validated RAG output models."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class Classification(BaseModel):
    category: Literal["FACTUAL", "BROADER"]
    reason: str = Field(min_length=1, max_length=300)
    confidence: float | None = Field(default=None, ge=0, le=1)


class Expansion(BaseModel):
    queries: list[str] = Field(min_length=3, max_length=3)

    @field_validator("queries")
    @classmethod
    def validate_queries(cls, queries: list[str]) -> list[str]:
        cleaned = [query.strip() for query in queries]
        if any(not query for query in cleaned) or len({q.casefold() for q in cleaned}) != 3:
            raise ValueError("Expected three non-empty, distinct search queries.")
        return cleaned


@dataclass
class Chunk:
    chunk_id: str
    text: str
    source: str
    page: int | None
    retrieval_score: float | None = None
    rerank_score: float | None = None
    initial_rank: int | None = None
    reranked_rank: int | None = None

    def as_dict(self) -> dict[str, object]:
        return vars(self).copy()

    @classmethod
    def from_dict(cls, value: dict[str, object]) -> "Chunk":
        return cls(**value)  # type: ignore[arg-type]


@dataclass
class Answer:
    question: str
    answer: str
    category: str | None
    cache_status: Literal["CACHE HIT", "CACHE MISS"]
    sources: list[Chunk] = field(default_factory=list)
    candidates: list[Chunk] = field(default_factory=list)
    expanded_queries: list[str] = field(default_factory=list)
    reason: str = ""
    confidence: float | None = None
    latency_seconds: float = 0
    timings: dict[str, float] = field(default_factory=dict)
    fallbacks: list[str] = field(default_factory=list)
    request_id: str = ""
    candidate_budget: int | None = None
    cache_match_question: str | None = None
    cache_similarity: float | None = None
    cache_ttl_seconds: float | None = None
    cached_at: str | None = None
    expires_at: str | None = None
