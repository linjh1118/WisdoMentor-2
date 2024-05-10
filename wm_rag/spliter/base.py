from typing import List
from abc import ABC, abstractmethod
from langchain_core.documents import Document

class Spliter(ABC):
    def __init__(self) -> None:
        super.__init__()
        return
    
    @abstractmethod
    def split_document(self, document: Document) -> List[Document]:
        pass