from typing import List
import time
from transformers import AutoTokenizer, BertModel
from torch import Tensor
from torch import no_grad

from models.EmbeddingGetter import EmbeddingGetter


class BertEmbeddingGetter(EmbeddingGetter):
    def _load_model(self) -> None:
        with self.lock:
            self.model = BertModel.from_pretrained(self.weight_path)
            self.tokenizer = AutoTokenizer.from_pretrained(self.weight_path)

    def get_embedding(self, sentences: List[str]) -> List[List[float]]:
        self._load_model()
        self._reset_timer()
        self.last_access_time = time.time()
        with self.lock:
            encoded_input = self.tokenizer(
                sentences, return_tensors="pt", padding=True, truncation=True
            )
            with no_grad():
                model_output = self.model(**encoded_input)
            res = model_output.pooler_output.tolist()
            return res
