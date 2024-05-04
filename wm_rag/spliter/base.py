from typing import List
from langchain_core.documents import Document

class Spliter:
    def __init__(self) -> None:
        return
    
    def split_document(self, document: Document) -> List[Document]:
        return [document]