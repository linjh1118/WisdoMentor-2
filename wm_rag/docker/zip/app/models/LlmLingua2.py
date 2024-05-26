from typing import List
from llmlingua import PromptCompressor
from .Zipper import Zipper


class LlmLingua2(Zipper):
    def __init__(self, weight_path: str) -> None:
        super().__init__(weight_path)

    def _load_model(self) -> None:
        with self.lock:
            if self.model is None:
                self.model = PromptCompressor(self.weight_path, use_llmlingua2=True)

    def zip(
        self, docs: List[str], query: str, rate: float = 0.5, instruction: str = ""
    ) -> str:
        self._load_model()
        self._reset_timer()
        compressed_prompt = self.model.compress_prompt(
            context=docs,
            question=query,
            instruction=instruction,
            rate=rate,
            concate_question=False,
        )
        return compressed_prompt["compressed_prompt"]
