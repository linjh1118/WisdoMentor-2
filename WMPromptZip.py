from llmlingua import PromptCompressor

class WMPromptZip:
    _model = ""
    _llm_lingua = None
    _llm_lingua_2 = None

    def __init__(self) -> None:
        pass
        
    def _del_llmlingua(self) -> None:
        if self._llm_lingua is not None:
            del self._llm_lingua
        if self._llm_lingua_2 is not None:
            del self._llm_lingua_2
    
    def _init_llmlingua(self, model: str) -> None:
        if self._model != model:
            self._del_llmlingua()
            self._llm_lingua = PromptCompressor(model_name=model)

    def _init_long_llmlingua(self, model: str) -> None:
        if self._model != model:
            self._del_llmlingua()
            self._llm_lingua = PromptCompressor(model_name=model)

    def _init_llmlingua_2(self, model: str) -> None:
        if self._model != model:
            self._del_llmlingua()
            self._llm_lingua_2 = PromptCompressor(model_name=model, 
                                                 use_llmlingua2=True)
    
    def get_model_name(self) -> str:
        return self._model

    def llmlingua_zip(self, model: str, 
                      context, 
                      instruction: str="", 
                      question: str="", 
                      rate: float=0.5,
                      force_tokens: list=[],
                      ) -> tuple:
        '''
        model: model name or path, model from huggingface or modelscope
        context: context of the prompt, str or list
        instruction: instruction of the prompt
        question: question of the prompt
        rate: compress rate of the context
        force_tokens: force tokens to keep in the context

        return: compressed prompt, compressed_tokens
        '''
        self._init_llmlingua(model)
        res = self._llm_lingua.compress_prompt(
            context=context,
            instruction=instruction,
            question=question,
            rate=rate,
            force_tokens=force_tokens,
        )

        return res['compressed_prompt'], res['compressed_tokens']
    
    def long_llmlingua_zip(self, model: str, 
                           context, 
                           instruction: str="", 
                           question: str="", 
                           rate: float=0.5,
                           dynamic_rate: float=0.3,
                           force_tokens: list=[],
                           ) -> tuple:
        '''
        model: model name or path, model from huggingface or modelscope
        context: context of the prompt, str or list
        instruction: instruction of the prompt
        question: question of the prompt
        rate: compress rate of the context
        dynamic_rate: ratio for dynamically adjusting context compression.
        force_tokens: force tokens to keep in the context

        return: compressed prompt, compressed_tokens
        '''
        self._init_llmlingua(model)
        res = self._llm_lingua.compress_prompt(
            context=context,
            instruction=instruction,
            question=question,
            rate=rate,
            force_tokens=force_tokens,
            dynamic_context_compression_ratio=dynamic_rate,
            condition_compare=True,
            context_budget="+100",
            rank_method="longllmlingua"
        )

        return res['compressed_prompt'], res['compressed_tokens']
    
    def llmlingua2_zip(self, model: str, 
                      context, 
                      instruction: str="", 
                      question: str="", 
                      rate: float=0.5,
                      force_tokens: list=[],
                      ) -> tuple:
        '''
        model: model name or path, model choose from ['llmlingua-2-bert-base-multilingual-cased-meetingbank', 'llmlingua-2-xlm-roberta-large-meetingbank']
        context: context of the prompt, str or list
        instruction: instruction of the prompt
        question: question of the prompt
        rate: compress rate of the context
        force_tokens: force tokens to keep in the context

        return: compressed prompt, compressed_tokens
        '''
        self._init_llmlingua_2(model)
        res = self._llm_lingua_2.compress_prompt(
            context=context,
            instruction=instruction,
            question=question,
            rate=rate,
            force_tokens=force_tokens,
        )

        return res['compressed_prompt'], res['compressed_tokens']
    