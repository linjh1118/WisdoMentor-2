from typing import List
from llama_index.core import Document
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from .base import Retrive

class ColbertRerankerRetrive(Retrive):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def recall(self, embed_query: List[float]) -> List[Document]:
        return
    
    def rerank(self, documents: List[Document]) -> List[Document]:
        return