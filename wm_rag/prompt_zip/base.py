from typing import List
from abc import ABC, abstractmethod
from langchain_core.documents import Document

class PromptZip(ABC):
    def __init__(self) -> None:
        super().__init__()
        return
    
    @abstractmethod
    def prompt_zip(self, prompt) -> str:
        raise NotImplementedError