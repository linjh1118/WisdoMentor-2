from typing import List
import asyncio

from langchain_core.documents import Document

from .base import WebExtractor
from entity.web_search_res import WebSearchRes
from web_extract.kimi.utils import Utils
from web_extract.kimi.conversation import Conversation


class KimiExtractor(WebExtractor):
    def __init__(self) -> None:
        super().__init__()
        return

    def extract(self, data: WebSearchRes) -> List[str]:
        res = asyncio.run(self.extract_async(data))
        return res

    async def extract_async(self, data: WebSearchRes) -> List[Document]:
        urls = data.urls
        await Utils().get_refresh_token()
        await Utils().get_access_token()
        extract_res = []
        for url in urls:
            prompt = f"请根据这个问题：{data.query}，总结这个链接的内容。{url}"
            conv = Conversation()
            await conv.create_conversation()
            response = await conv.do_conversation(
                [], [{"role": "user", "content": prompt}]
            )
            extract_res.append(Document(page_content=response["content"]))
        return extract_res
