from typing import List
import requests
from langchain_core.documents import Document

from .base import PromptZip


class LongLlmLinguaZipper(PromptZip):
    def __init__(self, router_path: str, port: str, path: str) -> None:
        super().__init__()
        if not router_path.startswith("http"):
            router_path = "http://" + router_path
        if path.startswith("/"):
            path = path[1:]
        self.router_path = router_path
        self.port = port
        self.path = path
        self.request_url = f"{self.router_path}:{self.port}/{self.path}"
        return

    def prompt_zip(
        self,
        prompt: List[Document],
        query: str,
        rate: float = 0.5,
        instruction: str = "",
    ) -> str:
        result = requests.post(
            url=f"{self.request_url}",
            json={
                "context": [doc.page_content for doc in prompt],
                "question": query,
                "rate": rate,
                "instruction": instruction,
            },
        )
        if result.status_code != 200:
            raise ValueError(
                f"Error in prompt_zip from {self.request_url} with status code {result.status_code}: {result.text}"
            )
        return result.json()["compressed_prompt"]
