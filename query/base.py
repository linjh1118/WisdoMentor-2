from typing import List
from abc import abstractmethod

class Query:
    def __init__(self) -> None:
        return
    
    @abstractmethod
    def query_extension(self, query: str) -> List[str]:
        """给定一个query, 调用各种方法，写出类似更到位的query，保证召回"""
    
    def query_rewrite(self, query: str) -> List[str]:
        return query