import os
import json
import random
from typing import Any, Dict, List
from langchain_core.documents import Document

from entity.evaluate_params import RecallEvaluateConfig
from .base import Evaluator
from store.ann_store import AnnStore
from embedding.bge_embedding import BgeEmbedding


class RecallEvaluator(Evaluator):
    def __init__(self, config_path: str) -> None:
        super().__init__(config_path)
        return

    def load_config(self) -> RecallEvaluateConfig:
        with open(self.config_path, "r") as f:
            config = json.load(f)
        config = self.init_config(config)
        return RecallEvaluateConfig(**config)

    def load_response(self) -> List[str]:
        self.response = None

    def init_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        config = super().init_config(config)
        config["query_json_path"] = os.path.join(
            os.path.dirname(self.config_path), config["query_json_path"]
        )
        config["response_json_path"] = config["query_json_path"]
        config["save_path"] = os.path.join(
            os.path.dirname(self.config_path), config.get("save_path", "")
        )
        if "max_tries" not in config:
            config["max_tries"] = 10
        return config

    def evaluate(self) -> None:
        ann_store = AnnStore(self.config.store_route_path, self.config.store_port)
        bge_embedding = BgeEmbedding(
            route_path=str(self.config.embedding_route_path),
            port=str(self.config.embedding_port),
        )
        embs = bge_embedding.embed_documents(
            [Document(page_content=doc) for doc in self.query]
        )
        ann_store.add_documents(
            [Document(page_content=doc) for doc in self.query], embs
        )

        hit = 0

        for i in range(self.config.max_tries):
            choosed_query = random.choice(self.query)
            sentences = choosed_query.split("。")
            query = random.choice(sentences)
            embs = bge_embedding.embed_documents([Document(page_content=query)])
            res = ann_store.search_by_embed(embs)
            for r in res:
                if r.page_content == choosed_query:
                    hit += 1
                    break

        recall = hit / self.config.max_tries

        with open(self.config.save_path, "w") as f:
            json.dump({"hit@10": recall}, f)
