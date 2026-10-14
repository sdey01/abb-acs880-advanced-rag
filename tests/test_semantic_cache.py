from src.schemas import Answer, Chunk
from src.semantic_cache import SemanticCache, cacheable


class FakeEmbeddings:
    def embed_query(self, text):
        if "different" in text:
            return [0.0, 1.0]
        return [1.0, 0.0]


def result(question="What is parameter 99.04?"):
    return Answer(
        question=question, answer="Parameter value is documented [manual.pdf, p. 4].",
        category="FACTUAL", cache_status="CACHE MISS",
        sources=[Chunk("1", "evidence", "manual.pdf", 4)],
    )


def test_exact_hit_and_similar_hit():
    cache = SemanticCache(FakeEmbeddings(), threshold=0.9)
    cache.set("What is parameter 99.04?", result())
    assert cache.lookup("What is parameter 99.04?")["similarity"] == 1
    assert cache.lookup("Please explain parameter 99.04") is not None


def test_technical_token_mismatch_is_rejected():
    cache = SemanticCache(FakeEmbeddings(), threshold=0.5)
    cache.set("What is parameter 99.04?", result())
    assert cache.lookup("What is parameter 99.05?") is None
    assert cache.lookup("What is parameter 99.04 in Hz?") is None
    cache.set("What is the maximum output 10 V?", result("What is the maximum output 10 V?"))
    assert cache.lookup("What is the maximum output 10 A?") is None
    assert cache.lookup("What is the maximum output 10%?") is None


def test_expiration_and_lazy_cleanup():
    cache = SemanticCache(FakeEmbeddings(), ttl=0)
    cache.set("question", result("question"))
    assert cache.lookup("question") is None
    assert cache.stats()["expired_removed"] == 1


def test_capacity_is_bounded():
    cache = SemanticCache(FakeEmbeddings(), capacity=2)
    for question in ("first", "second", "third"):
        cache.set(question, result(question))
    assert cache.stats()["entry_count"] == 2
    assert [e["question"] for e in cache.stats()["entries"]] == ["third", "second"]


def test_only_evidence_supported_success_is_cacheable():
    assert cacheable(result())
    unsupported = result()
    unsupported.answer = "The manual does not contain enough evidence."
    assert not cacheable(unsupported)
    unsupported.answer = "The manual does not specify the requested limit."
    assert not cacheable(unsupported)
    unsupported.answer = "The provided sources conflict about the requested parameter."
    assert not cacheable(unsupported)
    unsupported.answer = "I can't determine this from the excerpts provided."
    assert not cacheable(unsupported)
    unsupported.answer = "I can’t determine this from the excerpts provided."
    assert not cacheable(unsupported)
    unsupported.answer = "The excerpts do not provide a complete commissioning procedure."
    assert not cacheable(unsupported)
    unsupported = result()
    unsupported.sources = []
    assert not cacheable(unsupported)
