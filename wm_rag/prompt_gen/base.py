from typing import List
from abc import ABC, abstractmethod
from langchain_core.documents import Document

class PromptGen(ABC):
    def __init__(self) -> None:
        super().__init__()
        return
    
    @abstractmethod
    def prompt_gen(self, query, documents: List[Document]) -> str:
        raise NotImplementedError