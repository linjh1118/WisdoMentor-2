from typing import List
from abc import ABC, abstractmethod
from langchain_core.documents import Document


class PromptZip(ABC):
    def __init__(self) -> None:
        super().__init__()
        return

    @abstractmethod
    def prompt_zip(
        self,
        prompt: List[Document],
        query: str,
        rate: float = 0.5,
        instruction: str = "",
    ) -> str:
        raise NotImplementedError(
            "prompt_zip method must be implemented in a sub class."
        )
