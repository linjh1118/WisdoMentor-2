import os
from typing import Optional, Any
from llama_index.finetuning import SentenceTransformersFinetuneEngine
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset


class LLM2VecFinetuner:
    def __init__(
        self,
        weight_path: str,
        output_path: str,
        train_dataset_path: str,
        batch_size: int = 10,
        val_dataset_path: str = None,
        loss: Optional[Any] = None,
        epochs: int = 2,
        show_progress_bar: bool = True,
        evaluation_steps: int = 50,
        use_all_docs: bool = False,
    ) -> None:
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Model path {weight_path} does not exist.")
        if not os.path.exists(train_dataset_path):
            raise FileNotFoundError(
                f"Train dataset path {train_dataset_path} does not exist."
            )
        else:
            if not train_dataset_path.endswith(".json"):
                raise ValueError("Train dataset path should be a json file")
            else:
                self.train_dataset = EmbeddingQAFinetuneDataset.from_json(
                    train_dataset_path
                )
        if val_dataset_path is not None:
            if not os.path.exists(val_dataset_path):
                raise FileNotFoundError(
                    f"Val dataset path {val_dataset_path} does not exist."
                )
            else:
                if not val_dataset_path.endswith(".json"):
                    raise ValueError("Val dataset path should be a json file")
                else:
                    self.val_dataset = EmbeddingQAFinetuneDataset.from_json(
                        val_dataset_path
                    )
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        self.weight_path = weight_path
        self.output_path = output_path
        self.train_dataset_path = train_dataset_path
        self.val_dataset_path = val_dataset_path
        self.batch_size = batch_size
        self.loss = loss
        self.epochs = epochs
        self.show_progress_bar = show_progress_bar
        self.evaluation_steps = evaluation_steps
        self.use_all_docs = use_all_docs

    def finetune(self):
        finetune_engine = SentenceTransformersFinetuneEngine(
            self.train_dataset,
            self.weight_path,
            self.output_path,
            self.batch_size,
            self.val_dataset,
            self.loss,
            self.epochs,
            self.show_progress_bar,
            self.evaluation_steps,
            self.use_all_docs,
        )
        finetune_engine.finetune()