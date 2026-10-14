"""Generate exactly three distinct retrieval alternatives."""
from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from src.schemas import Expansion

SYSTEM = """Generate exactly three distinct alternative search queries for the ABB
ACS880 firmware manual. Do not answer. Preserve exact parameter identifiers,
numbers, units, quoted technical terms and conditions; do not change the meaning."""


class QueryExpander:
    def __init__(self, llm) -> None:
        self.llm = llm

    def expand(self, question: str) -> tuple[list[str], str | None]:
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM), ("human", "Expand this question:\n{question}")
        ])
        try:
            result = (prompt | self.llm.with_structured_output(Expansion)).invoke(
                {"question": question}
            )
            result = result if isinstance(result, Expansion) else Expansion.model_validate(result)
            return result.queries, None
        except Exception:
            return [], "Query expansion failed; retrieval used the original question only."
