import llmlingua
from .base import Prompt
from typing import List
from langchain_core.documents import Document

class llmlingua_zip(Prompt):
    def __init__(self, model_path: str) -> None:
        super().__init__()

    def prompt_zip(self, prompt) -> str:
        print(11111111111111111)
        
        return super().prompt_zip(prompt)
    
    def prompt_gen(self, query, documents: List[Document]) -> str:
        return super().prompt_gen(query, documents)