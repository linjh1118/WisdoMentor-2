import os
from transformers import AutoTokenizer, AutoModel
import torch


class BgeEmbeddingGetter:
    def __init__(
        self, model_path: str, tokenizer_path: str, enforce_cpu: bool = False
    ) -> None:
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer path {tokenizer_path} does not exist.")
        self.enforce_cpu = enforce_cpu
        self.device = (
            "cuda" if torch.cuda.is_available() and not self.enforce_cpu else "cpu"
        )
        try:
            self.model = AutoModel.from_pretrained(model_path).to(self.device)
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            raise e

    def get_embedding(self, sentences: list[str]) -> torch.Tensor:
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        ).to(self.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        del encoded_input
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )
        return sentence_embeddings
