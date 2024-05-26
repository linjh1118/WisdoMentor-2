from typing import List, Dict
from abc import ABC, abstractmethod

from langchain_core.documents import Document

from web_extract.base import WebExtractor


class WebSearcher(ABC):
    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def search(
        self, query: str, max_res: int = 3, extractor: WebExtractor = None
    ) -> List[Document]:
        raise NotImplementedError("Search method must be implemented by the subclass.")

    def search_queries(
        self,
        queries: List[str],
        max_res: List[int] = None,
        start_page: List[int] = None,
        end_page: List[int] = None,
    ) -> List[List[Document]]:
        if max_res is None:
            max_res = [3] * len(queries)
        if start_page is None:
            start_page = [1] * len(queries)
        if end_page is None:
            end_page = [1] * len(queries)
        assert len(queries) == len(max_res) == len(start_page) == len(end_page)
        return [
            self.search(
                query,
                max_res=max_res[i],
                start_page=start_page[i],
                end_page=end_page[i],
            )
            for i, query in enumerate(queries)
        ]
