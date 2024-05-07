from typing import List
from abc import ABC, abstractmethod

class Query(ABC):
    def __init__(self) -> None:
        super.__init__()
        return
    
    @abstractmethod
    def query_gen(self, query: str) -> List[str]:
        pass
    
    @abstractmethod
    def query_rewrite(self, query: str) -> List[str]:
        pass