from typing import List
import logging
import os

from langchain_core.documents import Document


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class RagLogger:
    def __init__(self) -> None:
        logging.basicConfig(
            level=logging.DEBUG,
            filename=os.path.join(os.path.dirname(__file__), "rag.log"),
            filemode="a",
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        )
        self.logger = logging.getLogger("rag_logger")

    def get_logger(self) -> logging.Logger:
        return self.logger
