import time, hashlib
from abc import ABC, abstractmethod

class Loader(ABC):
    def __init__(self) -> None:
        super().__init__()
        return

    def generate_doc_id(self) -> str:
        m = hashlib.md5(str(time.perf_counter()).encode("utf-8"))
        return m.hexdigest()
    
    @abstractmethod
    def load_file(self):
        raise NotImplementedError