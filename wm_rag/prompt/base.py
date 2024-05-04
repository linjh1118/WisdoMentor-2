from typing import List
from langchain_core.documents import Document

class Prompt:
    def __init__(self) -> None:
        return
    
    def prompt_gen(self, query, documents: List[Document]) -> str:
        return ""
    
    def prompt_zip(self, prompt) -> str:
        return prompt