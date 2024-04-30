import os
from sentence_transformers import SentenceTransformer
from sentence_transformers.evaluation import InformationRetrievalEvaluator
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset


class BgeEvaluator:
    def __init__(self, model_path: str, dataset_path: str, output_path: str):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model path {model_path} does not exist.")
        if not os.path.exists(dataset_path):
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist.")
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.model = SentenceTransformer(model_path)
        self.dataset_path = dataset_path
        self.output_path = output_path
        self.data_set = EmbeddingQAFinetuneDataset(self.dataset_path)
        self.evaluator = InformationRetrievalEvaluator(
            self.data_set.queries,
            self.data_set.corpus,
            self.data_set.relevant_docs,
            show_progress_bar=True,
        )

    def evaluate(self):
        self.evaluator(self.model, self.output_path)
