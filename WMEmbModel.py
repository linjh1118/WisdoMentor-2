import os
from typing import Optional, Any
from torch import cuda
from utils.bge.BgeEvaluator import BgeEvaluator
from utils.bge.BgeDataSpliter import BgeDataSpliter
from utils.bge.BgeVocabExpander import BgeVocabExpander
from utils.bge.BgeFinetuner import BgeFinetuner


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
                tokenizer=self.weight_path, data_file_path=data_file_path
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
