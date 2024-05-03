import os

class Recall_Evaluator:
    def __init__(self,model_type='bert'):
        self.model_type=model_type
        
        if self.model_type=='bert':
            import numpy as np
            from numpy.linalg import norm
            try:
                import sentence_transformers
            except ImportError:
                os.system("pip install sentence-transformers")
                import sentence_transformers

            self.model=self.model = sentence_transformers.SentenceTransformer('bert-base-nli-mean-tokens')
        
        elif self.model_type == 'use':
            import numpy as np
            from numpy.linalg import norm
            try:
                import tensorflow_hub as hub
            except ImportError:
                os.system("pip install tensorflow-hub")
                import tensorflow_hub as hub

            self.model = hub.load("https://tfhub.dev/google/universal-sentence-encoder/4")
        
        elif self.model_type == 'fasttext_en':
            try:
                import gensim
            except ImportError:
                os.system("pip install gensim")
                import gensim
            from gensim.models import KeyedVectors

            self.model = KeyedVectors.load_word2vec_format("https://dl.fbaipublicfiles.com/fasttext/vectors-english/wiki-news-300d-1M-subword.vec.zip",binary=False)
            
        elif self.model_type == 'fasttext_zh':
            try:
                import jieba
            except ImportError:
                os.system("pip install jieba")
                import jieba
            try:
                import gensim
            except ImportError:
                os.system("pip install gensim")
                import gensim
            from gensim.models import KeyedVectors

            self.model = KeyedVectors.load_word2vec_format("https://dl.fbaipublicfiles.com/fasttext/vectors-crawl/cc.zh.300.vec.gz",binary=False)


    def evaluate(self, question, res):
        if self.model_type == 'bert':
            return self._bert_evaluator(question, res)
        elif self.model_type == 'use':
            return self._use_evaluator(question, res)
        elif self.model_type == 'fasttext_en':
            return self._fasttext_evaluator_en(question, res)
        elif self.model_type == 'fasttext_zh':
            return self._fasttext_evaluator_zh(question, res)

    
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
    

    def _fasttext_evaluator_en(self, question, res):
        scores = []
        for dic in res:
            similarity = self.model.wv.n_similarity(question.split(), dic['text'].split())
            scores.append(similarity)
        return scores
    

    def _fasttext_evaluator_zh(self, question, res):
        scores = []
        for dic in res:
            similarity = self.model.wv.n_similarity(jieba.lcut(question), jieba.lcut(dic['text']))
            scores.append(similarity)
        return scores
    
class Generate_score:
    def __init__(self,model_type='bert'):
        self.model_type=model_type
        if self.model_type=='bert':
            import numpy as np
            from numpy.linalg import norm
            try:
                import sentence_transformers
            except ImportError:
                os.system("pip install sentence-transformers")
                import sentence_transformers

            self.model=self.model = sentence_transformers.SentenceTransformer('bert-base-nli-mean-tokens')

    def evaluate(self, question, res):
        if self.model_type == 'bert':
            return self._bert_evaluator(question, res)
        
    def _bert_evaluator(self, question, res):
        scores = []
        embeddings = self.model.encode([question, res])
        similarity = np.dot(embeddings[0], embeddings[1]) / (norm(embeddings[0]) * norm(embeddings[1]))
        scores.append(float(similarity))
        return scores