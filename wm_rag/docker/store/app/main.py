from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
from AnnStore import AnnoyStore


class DocEmbs(BaseModel):
    doc_list: List[str]
    doc_emb_list: List[List[float]]


class SearchParams(BaseModel):
    query_vec: List[List[float]]
    num: int = 50


class Docs(BaseModel):
    docs: List[str]


class Ids(BaseModel):
    ids: List[int]


app = FastAPI()


@app.post("/ann/add_docs")
def add_document(req: DocEmbs):
    doc_list = req.doc_list
    doc_emb_list = req.doc_emb_list
    store = AnnoyStore()
    store.add_documents(doc_list, doc_emb_list)


@app.post("/ann/search")
def search_by_embedding(req: SearchParams):
    query_vec = req.query_vec
    num = req.num
    store = AnnoyStore()
    res = store.search_by_embedding(query_vec, num)
    return {"knowledges": res}


@app.post("/ann/get_ids")
def get_id_by_docs(req: Docs):
    docs = req.docs
    store = AnnoyStore()
    ids = []
    for doc in docs:
        ids.append(store.get_id_by_doc(doc))
    return {"ids": ids}


@app.delete("/ann/remove_items")
def delete_by_ids(req: Ids):
    ids = req.ids
    store = AnnoyStore()
    for id in ids:
        store.delete_by_id(id)
