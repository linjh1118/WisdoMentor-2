from llmlingua import PromptCompressor
from .base import Prompt
from typing import List
from langchain_core.documents import Document

class LLMLingua2Zip(Prompt):
    """
    使用 LLMLingua2 进行压缩, 可选模型: 

    llmlingua-2-xlm-roberta-large-meetingbank
    
    llmlingua-2-bert-base-multilingual-cased-meetingbank (轻量)
    """
    _model: str = ""
    _llm_lingua2 = None
    rate: float = 0.5
    force_tokens: List[str] = []
    def __init__(self, model_path: str) -> None:
        super().__init__()
        self._model = model_path
        self._llm_lingua2 = PromptCompressor(model_name=model_path,
                                            use_llmlingua2=True)
    
    def get_model_name(self) -> str:
        return self._model

    def prompt_gen(self, query, documents: List[Document]) -> str:
        super().prompt_gen(query, documents)
        return None
    
    def prompt_zip(self, query: str, documents: List[Document]) -> str:
        doc_merge = ""
        for doc in documents:
            doc_merge += f"{doc.page_content}\n"
        
        zip_res = self._llm_lingua2.compress_prompt(
            context=doc_merge, 
            instruction="",
            question="",
            rate=self.rate,  
            force_tokens=self.force_tokens,
            concate_question=False,
            )
        doc_zip, zip_tokens = zip_res['compressed_prompt'], zip_res['compressed_tokens']
        
        question_merge = f"\n\n以上是有关于问题的参考资料，请根据上面的资料回答问题：{query}"
        
        zip_prompt = f"{doc_zip}{question_merge}"
        
        return zip_prompt