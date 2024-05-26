from typing import List
import time
from sentence_transformers import SentenceTransformer
from torch import Tensor

from models.EmbeddingGetter import EmbeddingGetter


class BgeEmbeddingGetter(EmbeddingGetter):
    def _load_model(self) -> None:
        with self.lock:
            self.model = SentenceTransformer(self.weight_path)

    def get_embedding(self, sentences: List[str]) -> List[List[float]]:
        self._load_model()
        self._reset_timer()
        self.last_access_time = time.time()
        with self.lock:
            tensors: list[Tensor] = self.model.encode(sentences)
            res: list[float] = [tensor.tolist() for tensor in tensors]
            return res
