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
    
    def add_documents(self, documents: List[Document], doc_embeds: List[List[float]]) -> bool:
        super().add_documents(documents, doc_embeds)
        requests.post(
            url=f"{self.request_url}/ann/add_docs",
            json={
                "doc_list": [doc.page_content for doc in documents],
                "doc_emb_list": doc_embeds 
            }
        )
        return True
    
    def search_by_embed(self, query_embed: List[float]) -> List[Document]:
        super().search_by_embed(query_embed)
        results = requests.post(
            url=f"{self.request_url}/ann/search",
            json={
                "query_vec": query_embed
            }
        ).json()["knowledges"]
        return [Document(page_content=knowledge) for knowledge in results]
    
    def delete_documents_by_ids(self, doc_ids: List[int]) -> bool:
        super().delete_documents_by_ids(doc_ids)
        requests.delete(
            url=f"{self.request_url}/ann/remove_items",
            json={
                "item_idx_list": doc_ids
            }
        )
        return True
    
        