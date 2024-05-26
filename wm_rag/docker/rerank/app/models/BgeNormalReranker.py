from typing import List
from models.Reranker import Reranker
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from torch import no_grad


class BgeNormalReranker(Reranker):
    def _load_model(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(self.weight_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.weight_path
        )
        self.model.eval()

    def rerank_documents(self, query: str, documents: List[str]) -> List[str]:
        self._load_model()
        self._reset_timer()
        pairs = [[query, doc] for doc in documents]
        with self.lock:
            with no_grad():
                inputs = self.tokenizer(
                    pairs,
                    padding=True,
                    truncation=True,
                    return_tensors="pt",
                    max_length=512,
                )
                scores = (
                    self.model(**inputs, return_dict=True)
                    .logits.view(
                        -1,
                    )
                    .float()
                    .tolist()
                )
            res = [(documents[i], scores[i]) for i in range(len(documents))]
            res.sort(key=lambda x: x[1], reverse=True)
            return [x[0] for x in res]
