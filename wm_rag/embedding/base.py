from typing import List
from abc import ABC, abstractmethod
from langchain_core.documents import Document

class Embedding(ABC):
    def __init__(self):
        super().__init__()
        return
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        pass
        
    @abstractmethod
    def embed_document(self, document: Document) -> List[float]:
        pass
    
    @abstractmethod
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        pass