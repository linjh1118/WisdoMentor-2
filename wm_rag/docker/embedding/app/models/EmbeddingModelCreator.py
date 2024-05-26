import os

from models.EmbeddingGetter import EmbeddingGetter
from models.BertEmbeddingGetter import BertEmbeddingGetter
from models.BgeEmbeddingGetter import BgeEmbeddingGetter


def get_embedding_model(type: str) -> EmbeddingGetter:
    if type == "bge":
        return BgeEmbeddingGetter(
            os.path.join(os.path.dirname(__file__), "../weights/bge-m3")
        )
    elif type == "bert":
        return BertEmbeddingGetter(
            os.path.join(os.path.dirname(__file__), "../weights/bert-chinese")
        )
    else:
        return ValueError("model type does not support!")
