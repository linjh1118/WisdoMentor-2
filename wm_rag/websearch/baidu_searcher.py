from typing import List, Dict
import requests

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from .base import WebSearcher
from entity.web_search_res import WebSearchRes
from web_extract.base import WebExtractor
from web_extract import TextExtractor


class BaiduSearcher(WebSearcher):
    def __init__(self) -> None:
        super().__init__()
        self.search_url = "http://www.baidu.com/s"
        return

    def _create_urls(
        self,
        query: str,
        max_res: int = 3,
        start_page: int = 1,
        end_page: int = 1,
    ) -> List[str]:
        urls = []
        for page in range(start_page, end_page + 1):
            params = {
                "wd": query,
                "rn": str(max_res),
                "pn": str((page - 1) * max_res),
            }
            response = requests.get(self.search_url, params=params)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, "html.parser")
                results = soup.find_all("h3", class_="t")
                for result in results:
                    link = result.find("a")
                    if link and link["href"].startswith("http"):
                        urls.append(link["href"])

        return urls

    def search(
        self,
        query: str,
        max_res: int = 3,
        extractor: WebExtractor = TextExtractor(),
        start_page: int = 1,
        end_page: int = 1,
    ) -> List[Document]:
        urls = self._create_urls(query, max_res, start_page, end_page)
        search_res = WebSearchRes(query=query, source="baidu", urls=[])
        for url in urls:
            search_res.urls.append(url)
        extract_res = extractor.extract(search_res)
        return extract_res
