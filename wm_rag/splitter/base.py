from typing import List
from abc import ABC, abstractmethod
from langchain_core.documents import Document

class Splitter(ABC):
    def __init__(self) -> None:
        super.__init__()
        return
    
    @abstractmethod
    def split_content(self, content: Document) -> List[Document]:
        raise  NotImplementedError