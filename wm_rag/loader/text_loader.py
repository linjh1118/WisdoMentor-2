from .base import Loader
from langchain_core.documents import Document

class TextLoader(Loader):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def load_file(self, text_content: str):
        return Document(page_content=text_content, metadata={"doc_id": self.generate_doc_id()})