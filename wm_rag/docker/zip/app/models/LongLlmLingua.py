from typing import List
from llmlingua import PromptCompressor
from .Zipper import Zipper


class LongLlmLingua(Zipper):
    def __init__(self, weight_path: str):
        super().__init__(weight_path)

    def _load_model(self) -> None:
        with self.lock:
            if self.model is None:
                self.model = PromptCompressor(self.weight_path)

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
            condition_compare=True,
            rank_method="longllmlingua",
            use_sentence_level_filter=False,
            context_budget="+200",
            dynamic_context_compression_ratio=0.4,
            reorder_context="sort",
            concate_question=False,
        )
        return compressed_prompt["compressed_prompt"]
