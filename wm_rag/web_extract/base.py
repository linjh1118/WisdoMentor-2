from abc import ABC, abstractmethod
from typing import List

from langchain_core.documents import Document

from entity.web_search_res import WebSearchRes


class WebExtractor(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def extract(self, data: WebSearchRes) -> List[Document]:
        raise NotImplementedError("Extract method must be implemented by the subclass.")
