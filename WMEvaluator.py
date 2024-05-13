import os
import typing
import numpy as np
from numpy.linalg import norm
import tensorflow_hub as hub
import sentence_transformers


class WMEvaluator:
    """当前默认bert"""
    def __init__(self,model_type='bert') -> None:
        self.model_type=model_type
        
        if self.model_type=='bert':
            self.model=self.model = sentence_transformers.SentenceTransformer('bert-base-nli-mean-tokens')
        
        elif self.model_type == 'use':
            self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")

    
    def recall_evaluator(self,question:str,res:list[dict]):
        if self.model_type == 'bert':
            return self._bert_evaluator(question, res)
        elif self.model_type == 'use':
            return self._use_evaluator(question, res)

    
    def generate_evaluator_by_bert(self,question:str,res:list[dict]):       #llm回答也组织成[{'text':'answer....'}]格式
        if self.model_type == 'bert':
            return self._bert_evaluator(question, res)
        elif self.model_type == 'use':
            return self._use_evaluator(question, res)

    
    
    
    def _bert_evaluator(self, question, res):
        scores = []
        for dic in res:
            embeddings = self.model.encode([question, dic['text']])
            similarity = np.dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
            scores.append(float(similarity))
        return scores

    
    def _use_evaluator(self, question, res):
        scores = []
        question_embedding = self.model([question])[0]
        for dic in res:
            text_embedding = self.model([dic['text']])[0]
            similarity = np.dot(question_embedding, text_embedding) / (
                        norm(question_embedding) * norm(text_embedding))
            scores.append(float(similarity))
        return scores

    

'''shili=WMEvaluator(model_type='use')
ans=shili.recall_evaluator("今年多大了？",[{'text':'刚满18岁~'}])
print(ans)'''
    

    