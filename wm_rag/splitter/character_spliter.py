from typing import List
from langchain_core.documents import Document
from langchain_text_splitters import CharacterTextSplitter
from .base import Splitter

class CharacterSplitter(Splitter):
    def __init__(
        self, 
        chunk_size: int, 
        separator: str = "\n", 
        chunk_overlap: int = 4
    ) -> None:
        super().__init__()
        self.splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        return
    
    def split_content(self, content: Document) -> List[Document]:
        return self.splitter.split_documents([content])