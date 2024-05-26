from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from models.BgeNormalReranker import BgeNormalReranker


class RerankRequest(BaseModel):
    query: str
    documents: List[str]


app = FastAPI()


@app.post("/bge_normal_rerank")
def bge_rerank(request: RerankRequest):
    query = request.query
    documents = request.documents
    reranker = BgeNormalReranker()
    res = reranker.rerank_documents(query, documents)
    return {"reranked_documents": res}
