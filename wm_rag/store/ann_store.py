import json
import requests
from .base import Store
from typing import List
from langchain_core.documents import Document


class AnnStore(Store):
    def __init__(self, router_path: str, port: str) -> None:
        super().__init__()
        self.router_path = router_path
        self.port = port
        self.request_url = f"{self.router_path}:{self.port}"
        return

    def add_documents(
        self, documents: List[Document], doc_embeds: List[List[float]]
    ) -> bool:
        super().add_documents(documents, doc_embeds)
        result = requests.post(
            url=f"{self.request_url}/ann/add_docs",
            json={
                "doc_list": [doc.page_content for doc in documents],
                "doc_emb_list": doc_embeds,
            },
        )
        if result.status_code != 200:
            raise ValueError(
                f"Error in add_documents from {self.request_url} with status code {result.status_code} : {result.text}"
            )
        return True

    def search_by_embed(
        self, query_embed: List[List[float]], k: int = 10
    ) -> List[Document]:
        super().search_by_embed(query_embed)
        result = requests.post(
            url=f"{self.request_url}/ann/search",
            json={"query_vec": query_embed, "num": k},
        )
        if result.status_code != 200:
            raise ValueError(
                f"Error in search by embedding from {self.request_url} with status code {result.status_code} : {result.text}"
            )
        results = result.json()["knowledges"][0]
        return [
            Document(page_content=knowledge) for knowledge in results
        ]  # 这里对Document的构造还需要再测试一下

    def get_id_by_docs(self, docs: List[Document]) -> List[int]:
        super().get_id_by_docs(docs)
        results = requests.post(
            url=f"{self.request_url}/ann/get_ids",
            json={"docs": [doc.page_content for doc in docs]},
        )
        if results.status_code != 200:
            raise Exception(f"Failed to get ids: {results.text}")
        return results.json()["ids"]

    def delete_documents_by_ids(self, doc_ids: List[int]) -> bool:
        super().delete_documents_by_ids(doc_ids)
        result = requests.delete(
            url=f"{self.request_url}/ann/remove_items", json={"ids": doc_ids}
        )
        if result.status_code != 200:
            raise ValueError(
                f"Error in delete documents from {self.request_url} with status code {result.status_code} : {result.text}"
            )
        return True
