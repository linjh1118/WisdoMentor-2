from typing import List
from langchain_core.documents import Document
from .base import Splitter
from langchain_text_splitters import RecursiveCharacterTextSplitter


class RecursiveSplitter(Splitter):
    def __init__(
        self,
        chunk_size: int,
        separators: List[str] = ["\n\n", "\n"],
        chunk_overlap: int = 4,
        length_function: callable = len,
    ) -> None:
        super().__init__()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function,
            is_separator_regex=False,
            separators=separators,
        )

    def split_content(self, content: Document) -> List[Document]:
        return self.text_splitter.create_documents([content.page_content])
