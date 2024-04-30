import os
from typing import Optional, Any
from torch import cuda
from torch import Tensor
from utils.bge.BgeEvaluator import BgeEvaluator
from utils.bge.BgeDataSpliter import BgeDataSpliter
from utils.bge.BgeVocabExpander import BgeVocabExpander
from utils.bge.BgeDataPreparer import BgeDataPreparer
from utils.bge.BgeFinetuner import BgeFinetuner
from utils.bge.BgeEmbeddingGetter import BgeEmbeddingGetter


class WMEmbModel:
    def __init__(self, weight_path) -> None:
        # return NotImplementedError
        self.device = "cuda" if cuda.is_available() else "cpu"
        self.weight_path = weight_path
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Model path {weight_path} does not exist.")

    def evaluate(self, val_dataset_path, evaluate_output_path) -> None:
        try:
            evaluator = BgeEvaluator(
                self.weight_path, val_dataset_path, evaluate_output_path
            )
        except FileNotFoundError as e:
            print(e)
            return
        except Exception as e:
            print(f"Error occurred when initializing BgeEvaluator: {e}")
            return
        evaluator.evaluate()

    def split(
        self,
        data_file_path: str,
        output_path: str,
        stop_words: list[str] = None,
        max_len: int = 512,
        train_size: float = 0.8,
    ):
        try:
            spliter = BgeDataSpliter(
                tokenizer_path=self.weight_path, data_file_path=data_file_path
            )
        except FileNotFoundError as e:
            print(e)
            return
        except Exception as e:
            print(f"Error occurred when initializing BgeDataSpliter: {e}")
            return

        spliter.split(output_path, stop_words, max_len, train_size)

    def expand(
        self, new_words: list[str], output_path: str, enforce_cpu: bool = False
    ) -> None:
        if new_words is None or len(new_words) == 0:
            print("No new words to expand.")
            return
        try:
            expander = BgeVocabExpander(self.weight_path, enforce_cpu)
        except FileNotFoundError as e:
            print(e)
            return
        except Exception as e:
            print(f"Error occurred when initializing BgeVocabExpander: {e}")
            return
        expander.expand(new_words, output_path)

    def prepare(self, train_file_path, val_file_path, output_dir):
        try:
            preparer = BgeDataPreparer(train_file_path, val_file_path, output_dir)
        except FileNotFoundError as e:
            print(e)
        except Exception as e:
            print(f"Error occurred when initializing BgeDataPreparer: {e}")
        preparer.prepare()

    def finetune(
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
        try:
            finetuner = BgeFinetuner(
                weight_path,
                output_path,
                train_dataset_path,
                batch_size,
                val_dataset_path,
                loss,
                epochs,
                show_progress_bar,
                evaluation_steps,
                use_all_docs,
            )
        except FileNotFoundError as e:
            print(e)
            return
        except ValueError as e:
            print(e)
            return
        except Exception as e:
            print(f"Error occurred when initializing BgeFinetuner: {e}")
            return
        finetuner.finetune()

    def get_embedding(
        self, model_path: str, tokenizer_path: str, sentences: list[str]
    ) -> Tensor:
        try:
            embedding_getter = BgeEmbeddingGetter(model_path, tokenizer_path)
        except FileNotFoundError as e:
            print(e)
            return
        except Exception as e:
            print(f"Error occurred when initializing BgeEmbeddingGetter: {e}")
            return
        return embedding_getter.get_embedding(sentences)
