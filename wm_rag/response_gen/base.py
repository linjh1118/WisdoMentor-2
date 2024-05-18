from typing import List
from abc import ABC, abstractmethod
from langchain_core.documents import Document

class ResponseGen(ABC):
    def __init__(self) -> None:
        super().__init__()
        return
    
    @abstractmethod
    def response_gen(self, prompt: str) -> str:
        raise NotImplementedError