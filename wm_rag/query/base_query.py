from typing import List
from .base import Query

class BaseQuery(Query):
    def __init__(self) -> None:
        super().__init__()
        return
    
    def query_gen(self, query: str) -> List[str]:
        super().query_gen(query)
        return [query]
    
    def query_rewrite(self, query: str) -> List[str]:
        super().query_rewrite(query)
        return query