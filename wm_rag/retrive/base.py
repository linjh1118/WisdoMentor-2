from typing import List
from abc import ABC, abstractmethod
from langchain_core.documents import Document

class Retrive(ABC):
    def __init__(self) -> None:
        super.__init__()
        return
    
    @abstractmethod
    def recall(self, embed_query: List[float]) -> List[Document]:
        pass
    
    @abstractmethod
    def rerank(self, documents: List[Document]) -> List[Document]:
        pass