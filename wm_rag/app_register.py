import os
import json
from typing import List
from langchain_core.documents import Document

from config import LOADER, SPLITTER, STORE, WEBSEARCH, PROMPTGEN, PROMPTZIP, RESPONSEGEN


class AppRegister:
    def __init__(self, app_name: str) -> None:
        if not os.path.exists("app_register_config.json"):
            raise OSError("The config file doesn't exist")
        with open("app_register_config.json", "r") as f:
            self.config = json.load(f)[app_name]
        self.database = [STORE[db_config["name"]](**db_config["args"]) for db_config in self.config["database"]]
        # loader 这里考虑是参数解析还是配置
        self.loader = LOADER[self.config["loader"]["name"]](**self.config["loader"]["args"])
        self.splitter = SPLITTER[self.config["splitter"]["name"]](**self.config["splitter"]["args"])
        self.websearch = [WEBSEARCH[web_config["name"]](**web_config["args"]) for web_config in self.config["websearch"]]
        self.prompt_gener = PROMPTGEN[self.config["prompt_gen"]["name"]](**self.config["prompt_gen"]["args"])
        self.prompt_zipper = PROMPTGEN[self.config["prompt_zip"]["name"]](**self.config["prompt_zip"]["args"])
        self.response_gener = PROMPTGEN[self.config["response_gen"]["name"]](**self.config["response_gen"]["args"])
        return
    
    def load_file(self, file_content: str):
        return self.loader.load_file(file_content)
    
    def split_content(self, content: Document) -> List[Document]:
        return self.splitter.split_content(content)
    
    def get_websearch_contents(self, query: str) -> List[Document]:
        contents = []
        for web in self.websearch:
            web.recall(query)
            contents.extend(web.llm_ans())
        return contents
    
    def get_prompt(self, query: str, contents: List[Document]) -> str:
        return self.prompt_gener.prompt_gen(query, contents)
    
    def zip_prompt(self, prompt: str) -> str:
        return self.prompt_zipper.prompt_zip(prompt)
    
    def get_response(self, prompt: str) -> str:
        return self.response_gener.response_gen(prompt)
