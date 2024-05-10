from typing import List
from abc import ABC, abstractmethod
from langchain_core.documents import Document

class Store(ABC):
    def __init__(self) -> None:
        super().__init__()
        return
    
    @abstractmethod
    def add_documents(self, document: List[Document], doc_embed: List[List[float]]) -> bool:
        pass
    
    @abstractmethod
    def search_by_embed(self, query_embed: List[float]) -> List[Document]:
        pass
    
    @abstractmethod
    def delete_documents_by_ids(self, doc_ids: List[int]) -> bool:
        pass