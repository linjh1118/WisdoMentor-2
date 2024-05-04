from typing import List
from langchain_core.documents import Document
from abc import abstractmethod

class Embedding:
    def __init__(self):
        return
    
    @abstractmethod
    def embed_text(self, text: str) -> List[float]:
        """return a embedding vec of text"""
    
    def embed_document(self, document: Document) -> List[float]:
        return self.embed_text(document.page_content)
    
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        return [self.embed_document(doc) for doc in documents]