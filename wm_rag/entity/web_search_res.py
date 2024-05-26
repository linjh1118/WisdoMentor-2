from pydantic import BaseModel
from typing import List, Dict


class WebSearchRes(BaseModel):
    query: str
    source: str
    urls: List[str]
