import os
import json
from typing import List
from langchain_core.documents import Document

from config import (
    LOADER,
    SPLITTER,
    EMBEDDING,
    STORE,
    WEBSEARCH,
    PROMPTGEN,
    PROMPTZIP,
    RESPONSEGEN,
    RERANK,
)


class AppRegister:
    def __init__(self, app_name: str) -> None:
        if not os.path.exists("app_register_config.json"):
            raise OSError("The config file doesn't exist")
        with open("app_register_config.json", "r") as f:
            self.config = json.load(f)[app_name]
        self.database = [
            STORE[db_config["name"]](**db_config["args"])
            for db_config in self.config["database"]
        ]
        # loader 这里考虑是参数解析还是配置
        self.loader = LOADER[self.config["loader"]["name"]](
            **self.config["loader"]["args"]
        )
        self.splitter = SPLITTER[self.config["splitter"]["name"]](
            **self.config["splitter"]["args"]
        )
        self.embedding = EMBEDDING[self.config["embedding"]["name"]](
            **self.config["embedding"]["args"]
        )
        self.websearch = [
            WEBSEARCH[web_config["name"]](**web_config["args"])
            for web_config in self.config["websearch"]
        ]
        self.reranker = RERANK[self.config["rerank"]["name"]](
            **self.config["rerank"]["args"]
        )
        self.prompt_gener = PROMPTGEN[self.config["prompt_gen"]["name"]](
            **self.config["prompt_gen"]["args"]
        )
        # self.prompt_zipper = PROMPTGEN[self.config["prompt_zip"]["name"]](**self.config["prompt_zip"]["args"])
        self.prompt_zipper = PROMPTZIP[self.config["prompt_zip"]["name"]](
            **self.config["prompt_zip"]["args"]
        )
        # self.response_gener = PROMPTGEN[self.config["response_gen"]["name"]](
        #     **self.config["response_gen"]["args"]
        # )
        self.response_gener = RESPONSEGEN[self.config["response_gen"]["name"]](
            **self.config["response_gen"]["args"]
        )
        return

    def add_database(self, docs: List[Document], embeds: List[List[float]]):
        for db in self.database:
            db.add_documents(docs, embeds)

    def load_file(self, file_content: str):
        return self.loader.load_file(file_content)

    def split_content(self, content: Document) -> List[Document]:
        return self.splitter.split_content(content)

    def get_embedding(self, query: str) -> List[float]:
        return self.embedding.embed_text(query)

    def get_embeddings(self, contents: List[Document]) -> List[List[float]]:
        return self.embedding.embed_documents(contents)

    def get_websearch_contents(self, query: str) -> List[Document]:
        contents = []
        for web in self.websearch:
            res = web.search(query)
            contents.extend(res)
        return contents

    def get_prompt(self, query: str, contents: List[Document]) -> str:
        return self.prompt_gener.prompt_gen(query, contents)

    def zip_prompt(self, query: str, prompt: List[Document]) -> str:
        return self.prompt_zipper.prompt_zip(prompt, query)

    def get_response(self, prompt: str) -> str:
        return self.response_gener.response_gen(prompt)

    def recall(self, query: List[str], query_embd: List[List[float]]) -> List[Document]:
        res = []
        for db in self.database:
            for i in range(len(query)):
                recall_res = db.search_by_embed([query_embd[i]])
                rerank_res = self.reranker.rerank_documents(query[i], recall_res)[:30]
                res.extend(rerank_res)
        return res
