import pytest

from src.schemas import Classification, Expansion


def test_classification_accepts_only_supported_categories():
    assert Classification(category="FACTUAL", reason="Direct fact").category == "FACTUAL"
    with pytest.raises(ValueError):
        Classification(category="OTHER", reason="Invalid")


def test_expansion_requires_three_distinct_queries():
    assert len(Expansion(queries=["one", "two", "three"]).queries) == 3
    with pytest.raises(ValueError):
        Expansion(queries=["one", "one", "three"])
