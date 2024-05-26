from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from models.EmbeddingModelCreator import get_embedding_model

app = FastAPI()


class TextRequest(BaseModel):
    text: List[str]
    model_type: str


@app.post("/get_embedding")
def process(request: TextRequest):
    texts = request.text
    model_type = request.model_type
    embedding_getter = get_embedding_model(model_type)
    results = embedding_getter.get_embedding(texts)
    return {"embedding": results}
