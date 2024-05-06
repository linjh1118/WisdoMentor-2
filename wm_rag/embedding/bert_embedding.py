from langchain_core.documents import Document
import requests
from typing import List
from .base import Embedding

class BertEmbedding(Embedding):
    def __init__(self, route_path: str, port: str):
        super().__init__()
        self.route_path = route_path
        self.port = port
        self.request_url = f"{self.route_path}:{self.port}"
        return
    
    def embed_text(self, text: str) -> List[float]:
        super().embed_text(text)
        result = requests.post(
            url=f"{self.request_url}/get_embedding/",
            json={
                "text": text
            }
        )
        return result.json()["embedding"]
    
    def embed_document(self, document: Document) -> List[float]:
        super().embed_document(document)
        return self.embed_text(document.page_content)
    
    def embed_documents(self, documents: List[Document]) -> List[List[float]]:
        super().embed_documents(documents)
        return [self.embed_text(doc.page_content) for doc in documents]