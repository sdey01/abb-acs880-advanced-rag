from src.classifier import QuestionClassifier
from src.query_expansion import QueryExpander
from src.reranker import CrossEncoderReranker
from src.schemas import Chunk


class BrokenLLM:
    def with_structured_output(self, schema):
        raise RuntimeError("model unavailable")


def test_classifier_uses_visible_safe_fallback():
    result, used_fallback = QuestionClassifier(BrokenLLM()).classify(
        "Explain how the two settings relate"
    )
    assert used_fallback
    assert result.category == "BROADER"


def test_expansion_failure_returns_warning_and_no_fake_expansions():
    queries, warning = QueryExpander(BrokenLLM()).expand("What is parameter 99.04?")
    assert queries == []
    assert warning


def test_cross_encoder_reranks_by_score(monkeypatch):
    class FakeModel:
        def predict(self, pairs, show_progress_bar=False):
            return [0.2, 0.9]

    monkeypatch.setattr("src.reranker.CrossEncoder", lambda *args, **kwargs: FakeModel())
    reranker = CrossEncoderReranker()
    candidates = [
        Chunk("first", "low relevance", "manual.pdf", 1),
        Chunk("second", "high relevance", "manual.pdf", 2),
    ]
    ranked = reranker.rerank("question", candidates)
    assert [chunk.chunk_id for chunk in ranked] == ["second", "first"]
    assert ranked[0].reranked_rank == 1
