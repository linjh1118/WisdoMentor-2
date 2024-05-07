from typing import List
from langchain_core.documents import Document
from .base import Retrive

class BaseRetrive(Retrive):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def recall(self, embed_query: List[float]) -> List[Document]:
        super().recall(embed_query)
        pass
    
    def rerank(self, documents: List[Document]) -> List[Document]:
        super().rerank(documents)
        return documents