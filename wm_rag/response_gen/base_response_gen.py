# import ollama
from typing import List
from langchain_core.documents import Document
from .base import ResponseGen
import requests


class BaseResponseGen(ResponseGen):
    def __init__(self, router_path: str, port: str) -> None:
        super().__init__()
        self.router_path = router_path
        self.port = port
        self.request_url = f"{self.router_path}:{self.port}"
        return

    def response_gen(self, prompt: str) -> str:
        res = requests.post(
            url=f"{self.request_url}/ollama_generate",
            json={"model_name": "qwen:14b", "text": prompt},
        ).json()["res"]
        return res
