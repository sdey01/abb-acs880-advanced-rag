from src.rag_pipeline import RagPipeline
from src.schemas import Answer, Chunk, Classification


class CacheMiss:
    def lookup(self, question):
        return None

    def set(self, question, answer):
        from datetime import datetime, timedelta, timezone
        return datetime.now(timezone.utc) + timedelta(hours=1)

    def record(self, event):
        pass


class FakeLLM:
    def invoke(self, messages):
        class Response:
            content = "A grounded answer [manual.pdf, p. 1]."
        return Response()


class FakeClassifier:
    category = "FACTUAL"

    def __init__(self, llm):
        self.category = type(self).category

    def classify(self, question):
        return Classification(category=self.category, reason="Test category"), False


class FakeExpander:
    def __init__(self, llm):
        pass

    def expand(self, question):
        return ["parameter detail", "parameter definition", "manual parameter"], None


class FakeIndex:
    def __init__(self):
        self.budget = None

    def retrieve(self, queries, budget):
        self.budget = budget
        return [Chunk("1", "manual evidence", "manual.pdf", 1)]


class FakeReranker:
    def rerank(self, question, candidates):
        return candidates


def test_factual_pipeline_uses_five_candidates(monkeypatch):
    monkeypatch.setattr("src.rag_pipeline.QuestionClassifier", FakeClassifier)
    monkeypatch.setattr("src.rag_pipeline.QueryExpander", FakeExpander)
    index = FakeIndex()
    pipeline = RagPipeline(index, CacheMiss(), FakeReranker(), FakeLLM())
    answer = pipeline.ask("What is parameter 99.04?")
    assert index.budget == 5
    assert answer.category == "FACTUAL"
    assert len(answer.expanded_queries) == 3


def test_broader_budget(monkeypatch):
    monkeypatch.setattr("src.rag_pipeline.QuestionClassifier", FakeClassifier)
    monkeypatch.setattr("src.rag_pipeline.QueryExpander", FakeExpander)
    index = FakeIndex()
    pipeline = RagPipeline(index, CacheMiss(), FakeReranker(), FakeLLM())
    monkeypatch.setattr(FakeClassifier, "category", "BROADER")
    pipeline.ask("Explain how these settings relate")
    assert index.budget == 20


def test_semantic_cache_hit_skips_llm_and_retrieval():
    cached_chunk = Chunk("cached", "stored evidence", "manual.pdf", 3)

    class HitCache(CacheMiss):
        def lookup(self, question):
            from datetime import datetime, timedelta, timezone
            class Entry:
                answer = "Previously grounded answer [manual.pdf, p. 3]."
                category = "FACTUAL"
                question = "cached question"
                sources = [cached_chunk.as_dict()]
                created = datetime.now(timezone.utc)
                expires = created + timedelta(minutes=30)
            return {"entry": Entry(), "similarity": .97, "remaining_ttl": 1800}

    class MustNotRetrieve(FakeIndex):
        def retrieve(self, queries, budget):
            raise AssertionError("Cache hit should skip retrieval.")

    pipeline = RagPipeline(MustNotRetrieve(), HitCache(), FakeReranker(), llm=None)
    answer = pipeline.ask("similar cached question")
    assert answer.cache_status == "CACHE HIT"
    assert answer.cache_similarity == .97
    assert answer.sources[0].page == 3
