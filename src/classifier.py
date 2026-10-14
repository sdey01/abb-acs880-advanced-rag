"""Validated model-based classification with safe deterministic fallback."""
from __future__ import annotations

import re

from langchain_core.prompts import ChatPromptTemplate

from src.schemas import Classification

SYSTEM = """Classify only; do not answer. FACTUAL is a narrow lookup of a specific
parameter, value, definition, setting, limit or identifier. BROADER is an
explanation, multi-part question, comparison, relationship, procedure or synthesis."""


class QuestionClassifier:
    def __init__(self, llm) -> None:
        self.llm = llm

    def classify(self, question: str) -> tuple[Classification, bool]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM), ("human", "Classify this question only:\n{question}")
        ])
        try:
            result = (prompt | self.llm.with_structured_output(Classification)).invoke(
                {"question": question}
            )
            return (result if isinstance(result, Classification)
                    else Classification.model_validate(result)), False
        except Exception:
            broad = re.search(
                r"\b(explain|compare|comparison|relationship|how do|how does|steps|"
                r"procedure|summari[sz]e|differences|multiple)\b",
                question, re.I,
            ) or question.count("?") > 1
            return Classification(
                category="BROADER" if broad else "FACTUAL",
                reason="Deterministic fallback used because model classification was unavailable.",
            ), True
