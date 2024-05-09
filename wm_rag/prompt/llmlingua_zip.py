from llmlingua import PromptCompressor
from .base import Prompt
from typing import List
from langchain_core.documents import Document

class LLMLinguaZip(Prompt):
    """
    使用 LongLLMLingua 进行压缩
    可选模型来自 huggingface 或 modelscope
    """
    _model: str = ""
    _llm_lingua = None
    rate: float = 0.5
    dynamic_rate: float = 0.3
    force_tokens: List[str] = []
    def __init__(self, model_path: str) -> None:
        super().__init__()
        self._model = model_path
        self._llm_lingua = PromptCompressor(model_name=model_path)
    
    def get_model_name(self) -> str:
        return self._model

    def prompt_gen(self, query, documents: List[Document]) -> str:
        super().prompt_gen(query, documents)
        return None
    
    def prompt_zip(self, query: str, documents: List[Document]) -> str:
        doc_merge = ""
        for doc in documents:
            doc_merge += f"{doc.page_content}\n"
        
        zip_res = self._llm_lingua.compress_prompt(
            context=doc_merge, 
            instruction="",
            question=query,
            rate=self.rate, 
            dynamic_context_compression_ratio=self.dynamic_rate, 
            force_tokens=self.force_tokens,
            condition_compare=True,
            context_budget="+100",
            rank_method="longllmlingua",
            concate_question=False,
            )
        doc_zip, zip_tokens = zip_res['compressed_prompt'], zip_res['compressed_tokens']
        
        question_merge = f"\n以上是有关于问题的参考资料，请根据上面的资料回答问题：{query}"
        
        zip_prompt = f"{doc_zip}{question_merge}"
        
        return zip_prompt