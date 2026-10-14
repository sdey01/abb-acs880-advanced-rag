"""Bounded in-memory semantic cache with technical-token checks and TTL."""
from __future__ import annotations

import re
import threading
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

from config import CACHE_MAX_ENTRIES, CACHE_THRESHOLD, CACHE_TTL_SECONDS
from src.schemas import Answer


@dataclass
class Entry:
    question: str
    normalized: str
    vector: list[float]
    answer: str
    sources: list[dict[str, object]]
    category: str | None
    created: datetime
    expires: datetime


class SemanticCache:
    def __init__(self, embedder, ttl: int = CACHE_TTL_SECONDS,
                 capacity: int = CACHE_MAX_ENTRIES, threshold: float = CACHE_THRESHOLD) -> None:
        self.embedder, self.ttl, self.capacity = embedder, ttl, capacity
        self.threshold = threshold
        self.entries: list[Entry] = []
        self.lock = threading.RLock()
        self.expired_removed = 0
        self.activity = deque(maxlen=20)

    @staticmethod
    def normalize(text: str) -> str:
        return " ".join(text.casefold().split())

    @staticmethod
    def technical_tokens(text: str) -> set[str]:
        identifiers = re.findall(r"\b(?=[A-Za-z0-9._/-]*\d)[A-Za-z0-9][A-Za-z0-9._/-]*\b", text)
        measurements = re.findall(
            r"(?<![A-Za-z0-9])\d+(?:\.\d+)?\s*(?:%|rpm|khz|mhz|hz|mv|kv|ma|"
            r"kw|mw|kwh|ms|sec(?:onds?)?|minutes?|hours?|bar|kpa|pa|v|a|"
            r"°\s*[cf]|celsius|degrees?)(?![A-Za-z])",
            text, re.I,
        )
        units = re.findall(
            r"\b(?:rpm|khz|mhz|hz|mv|kv|ma|kw|mw|kwh|ms|seconds?|minutes?|"
            r"hours?|bar|kpa|pa|celsius|degrees?)\b|%|°\s*[cf]\b",
            text, re.I,
        )
        return {token.casefold() for token in identifiers + measurements + units}

    @classmethod
    def tokens_match(cls, left: str, right: str) -> bool:
        return cls.technical_tokens(left) == cls.technical_tokens(right)

    @staticmethod
    def cosine(left: list[float], right: list[float]) -> float:
        if not left or len(left) != len(right):
            return 0
        dot = sum(a*b for a, b in zip(left, right))
        norm = (sum(x*x for x in left) * sum(x*x for x in right)) ** .5
        return dot/norm if norm else 0

    def purge_expired(self) -> int:
        now = datetime.now(timezone.utc)
        with self.lock:
            count = len(self.entries)
            self.entries = [item for item in self.entries if item.expires > now]
            removed = count - len(self.entries)
            self.expired_removed += removed
            return removed

    def lookup(self, question: str):
        self.purge_expired()
        normalized, best, vector = self.normalize(question), None, None
        with self.lock:
            entries = list(self.entries)
        for item in entries:
            if not self.tokens_match(question, item.question):
                continue
            if normalized == item.normalized:
                score = 1.0
            else:
                if vector is None:
                    vector = self.embedder.embed_query(question)
                score = self.cosine(vector, item.vector)
            remaining = (item.expires - datetime.now(timezone.utc)).total_seconds()
            if remaining > 0 and score >= self.threshold and (best is None or score > best["similarity"]):
                best = {"entry": item, "similarity": score, "remaining_ttl": remaining}
        return best

    def set(self, question: str, result: Answer) -> datetime:
        self.purge_expired()
        now = datetime.now(timezone.utc)
        expiry = now + timedelta(seconds=self.ttl)
        item = Entry(question, self.normalize(question), self.embedder.embed_query(question),
                     result.answer, [chunk.as_dict() for chunk in result.sources],
                     result.category, now, expiry)
        with self.lock:
            self.entries = [old for old in self.entries if old.normalized != item.normalized]
            self.entries.append(item)
            if len(self.entries) > self.capacity:
                del self.entries[:len(self.entries) - self.capacity]
        return expiry

    def stats(self) -> dict[str, object]:
        self.purge_expired()
        with self.lock:
            return {
                "entry_count": len(self.entries), "max_entries": self.capacity,
                "ttl_seconds": self.ttl, "expired_removed": self.expired_removed,
                "activity": list(self.activity),
                "entries": [{"question": e.question, "created_at": e.created.isoformat(),
                             "expires_at": e.expires.isoformat()} for e in reversed(self.entries)],
            }

    def record(self, event: dict[str, object]) -> None:
        with self.lock:
            self.activity.appendleft(event)


def cacheable(answer: Answer) -> bool:
    if not answer.answer.strip() or not answer.sources:
        return False
    unsupported = re.compile(
        r"(?:manual\s+(?:does not|doesn't|doesnt)\s+(?:contain|specify|state|provide)"
        r"|manual\s+does\s+not\s+provide\s+(?:a\s+)?complete"
        r"|(?:manual|excerpts?|evidence)\s+(?:do|does)\s+not\s+(?:include|contain|provide|cover|show)"
        r"|not\s+(?:found|stated|specified|provided)\s+in\s+the\s+manual"
        r"|(?:not enough|insufficient)\s+(?:information|evidence)"
        r"|(?:cannot|unable to)\s+determine"
        r"|can[^\w]*t\s+determine"
        r"|do not provide enough|don't provide enough"
        r"|not included here|not supplied here|not provided in the excerpts"
        r"|no excerpt for|not covered by the excerpts"
        r"|sources?\s+(?:are\s+)?conflict(?:ing)?|conflicting\s+(?:evidence|information))",
        re.IGNORECASE,
    )
    return unsupported.search(answer.answer) is None
