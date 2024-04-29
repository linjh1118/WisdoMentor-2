from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor
from llama_index.postprocessor.colbert_rerank import ColbertRerank
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llama_index.core.postprocessor import SentenceTransformerRerank
from llama_index.core.postprocessor import LongContextReorder
from llama_index.llms.ollama import Ollama

class WMReranker:
    def __init__(self, retriever, top_n=5, llm_model="gemma:2b", llm_temperature=0.1):
        self.retriever = retriever
        self.llm = Ollama(model=llm_model, temperature=llm_temperature)
        self.colbert_reranker = ColbertRerank(
            top_n=top_n,
            model="colbert-ir/colbertv2.0",
            tokenizer="colbert-ir/colbertv2.0",
            keep_retrieval_score=True,
        )
        self.flag_reranker = FlagEmbeddingReranker(
            model="BAAI/bge-reranker-large", 
            top_n=top_n
        )
        self.gpt_reranker = RankGPTRerank(
            llm=self.llm,
            top_n=top_n,
            verbose=True
        )
        self.st_reranker = SentenceTransformerRerank(
            model="cross-encoder/ms-marco-MiniLM-L-2-v2",
            top_n=top_n
        )

    #Colbert Rerank
    def colbert_rerank(self, question, similarity_cutoff=0.6):
        query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=None,  # 如果是None，自动采用设置的llm判断相似度
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff), self.colbert_reranker],  # 添加重排器
        )
        reranked_response = query_engine.query(question)
        return reranked_response

    #FlagEmbedding Reranker
    def flag_rerank(self, question, similarity_cutoff=0.6):
        query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=None,  # 如果是None，自动采用设置的llm判断相似度
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff), self.flag_reranker],  # 添加重排器
        )
        reranked_response = query_engine.query(question)
        return reranked_response
    #Rank GPTReranker
    def gpt_rerank(self, question, similarity_cutoff=0.6):
        query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=None,  # 如果是None，自动采用设置的llm判断相似度
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff), self.gpt_reranker],  # 添加重排器
        )
        reranked_response = query_engine.query(question)
        return reranked_response

    #SentenceTransformerRerank
    def st_rerank(self, question, similarity_cutoff=0.6):
        query_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=None,  # 如果是None，自动采用设置的llm判断相似度
            node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff), self.st_reranker],  # 添加重排器
        )
        reranked_response = query_engine.query(question)
        return reranked_response
        
    #longcontext Rerank
    def long_context_rerank(self, question):
        self.reorder_engine = RetrieverQueryEngine(
            retriever=self.retriever,
            response_synthesizer=None,  #如果是None，自动采用设置的llm判断相似度
            node_postprocessors=[self.reorder],
            )
        reranked_response = self.reorder_engine.query(question)
        return reranked_response

