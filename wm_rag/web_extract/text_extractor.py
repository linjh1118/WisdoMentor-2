from typing import List
import requests

from bs4 import BeautifulSoup
from langchain_core.documents import Document

from .base import WebExtractor
from entity.web_search_res import WebSearchRes


class TextExtractor(WebExtractor):
    def __init__(self) -> None:
        super().__init__()
        return

    def extract(self, data: WebSearchRes) -> List[Document]:
        assert (
            data.source != "arxiv"
        ), f"source: {data.source} is not supported for text extractor."
        urls = data.urls
        extract_res = []
        for url in urls:
            # TODO: 适配需要JavaScript渲染的网页
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content, "html.parser")
                extract_res.append(Document(page_content=soup.get_text()))
            else:
                # TODO: 查看是否有除了JavaScript渲染外的其他原因导致的失败
                extract_res.append(Document(page_content=""))
        return extract_res
