import os
import torch
from transformers import AutoModel, AutoTokenizer
from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from base import *


app = FastAPI()


class TextInput(BaseModel):
    text: str


class BGEModel:
    _instance = None

    def __new__(
        cls, model_path: str, tokenizer_path: str, device: str = 'auto', **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.initialize(model_path, tokenizer_path, device)
        return cls._instance

    def initialize(self, model_path, tokenizer_path, device):
        self.model = AutoModel.from_pretrained(model_path).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

    def get_embedding(self, sentences: list[str]) -> torch.Tensor:
        if not isinstance(sentences, list):
            sentences = [sentences]
        encoded_input = self.tokenizer(
            sentences, padding=True, truncation=True, return_tensors="pt"
        ).to(self.model.device)
        with torch.no_grad():
            model_output = self.model(**encoded_input)
            sentence_embeddings = model_output[0][:, 0]
        del encoded_input
        sentence_embeddings = torch.nn.functional.normalize(
            sentence_embeddings, p=2, dim=1
        )
        sentence_embeddings = sentence_embeddings.cpu().numpy().tolist()
        return sentence_embeddings[0]


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='/home/chy/dream/LLMs/bge-large-zh-v1.5')
    parser.add_argument('--tokenizer_path', type=str, default='/home/chy/dream/LLMs/bge-large-zh-v1.5')
    parser.add_argument('--device', type=str, default='auto')
    parser.add_argument('--port', type=int, default=8888)
    args = parser.parse_args()
    args.device = set_device(args.device)
    return args

args = get_args()
logger = set_naive_logger('../log_cache')
log_hyperparams(args)

bge_model = BGEModel(**vars(args))

@app.post("/get_embedding/")
def embed_text(text_input: TextInput):
    embedding = bge_model.get_embedding(text_input.text)
    return {"embedding": embedding}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8888)
    


