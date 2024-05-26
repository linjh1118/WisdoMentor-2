from typing import List, Dict

import arxiv
from langchain_core.documents import Document

from .base import WebSearcher
from entity.web_search_res import WebSearchRes
from web_extract.base import WebExtractor
from web_extract import PdfExtractor


class ArxivSearcher(WebSearcher):
    def __init__(self) -> None:
        super().__init__()

    def search(
        self,
        query: str,
        max_res: int = 3,
        start_page: int = -1,
        end_page: int = -1,
        extractor: WebExtractor = PdfExtractor(),
    ) -> List[Document]:
        search_res = WebSearchRes(query=query, source="arxiv", urls=[])
        sorter = arxiv.SortCriterion.Relevance
        searcher = arxiv.Search(query, max_results=max_res, sort_by=sorter)
        for res in arxiv.Client().results(searcher):
            search_res.urls.append(res.pdf_url)
        extract_res = extractor.extract(search_res)
        return extract_res
