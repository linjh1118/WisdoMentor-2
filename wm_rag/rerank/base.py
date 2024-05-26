from abc import ABC, abstractmethod
from typing import List
from langchain_core.documents import Document


class Reranker(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        return NotImplementedError(
            "rerank_documents method must be implemented in a sub class."
        )
