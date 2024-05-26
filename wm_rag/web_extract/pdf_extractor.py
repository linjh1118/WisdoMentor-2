from typing import List
import os
import requests

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from .base import WebExtractor
from entity.web_search_res import WebSearchRes


class PdfExtractor(WebExtractor):
    def __init__(self) -> None:
        super().__init__()
        return

    def extract(self, data: WebSearchRes) -> List[Document]:
        assert (
            data.source == "arxiv"
        ), f"source: {data.source} is not supported for pdf extractor."
        urls = data.urls
        extract_res = []
        if not os.path.exists(os.path.join(os.path.dirname(__file__), "pdfs")):
            os.makedirs(os.path.join(os.path.dirname(__file__), "pdfs"))
        for url in urls:
            response = requests.get(url)
            if response.status_code != 200:
                extract_res.append("")
                continue
            pdf_path = os.path.join(
                os.path.dirname(__file__), "pdfs", url.split("/")[-1]
            )
            with open(pdf_path, "wb") as f:
                f.write(response.content)
            loader = PyPDFLoader(pdf_path)
            extract_res += loader.load()
            os.remove(pdf_path)
        return extract_res
