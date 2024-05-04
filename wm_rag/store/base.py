from typing import List
from langchain_core.documents import Document

class Store:
    def __init__(self) -> None:
        return
    
    def add_document(self, document: Document, doc_embed: List[float]) -> bool:
        return True
    
    def search_by_embed(self, query_embed: List[float]) -> List[Document]:
        return []
    
    def delete_document_by_id(self, doc_id: str) -> bool:
        return True