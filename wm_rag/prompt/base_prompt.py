# import ollama
from typing import List
from langchain_core.documents import Document
from .base import Prompt
import requests

class BasePrompt(Prompt):
    def __init__(self, router_path: str, port: str) -> None:
        super().__init__()
        self.router_path = router_path
        self.port = port
        self.request_url = f"{self.router_path}:{self.port}"
        return
    
    def prompt_gen(self, query, documents: List[Document]) -> str:
        super().prompt_gen(query, documents)
        prompt_merge = ""
        for doc in documents:
            prompt_merge += f"{doc.page_content}\n"
        prompt_merge += f"以上是有关于问题的参考资料，请根据上面的资料回答问题：{query}"
        # results = ollama.chat(model="qwen:7b", messages=[
        #     {
        #         'role': 'user',
        #         'content': prompt_merge,
        #     },
        # ])
        # return results['message']['content']
        res = requests.post(
            url=f"{self.request_url}/ollama_generate",
            json={
                "model_name": 'qwen:14b',
                "text": prompt_merge
            }
        ).json()["res"]
        
        return res
    
    def prompt_zip(self, prompt) -> str:
        super().prompt_zip(prompt)
        return prompt