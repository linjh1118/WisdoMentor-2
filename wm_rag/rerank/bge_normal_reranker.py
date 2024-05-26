from typing import List
import requests
from langchain_core.documents import Document

from .base import Reranker


class BgeNormalReranker(Reranker):
    def __init__(self, router_path: str, port: str, path: str):
        super().__init__()
        if path.startswith("/"):
            path = path[1:]
        self.route_path = router_path if router_path.startswith("http") else "http://"
        self.port = port
        self.request_url = f"{router_path}:{port}/{path}"

    def rerank_documents(self, query: str, docs: List[Document]) -> List[Document]:
        result = requests.post(
            url=self.request_url,
            json={"query": query, "documents": [doc.page_content for doc in docs]},
        )
        if result.status_code != 200:
            raise ValueError(
                f"Error in rerank_documents from {self.request_url} with status code {result.status_code}: {result.text}"
            )
        return [
            Document(page_content=doc) for doc in result.json()["reranked_documents"]
        ]
