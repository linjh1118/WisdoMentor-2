import os
import json

from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
from transformers import AutoTokenizer

from llama_index.core.node_parser import (
    SentenceSplitter,
)

from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor

from llama_index.postprocessor.colbert_rerank import ColbertRerank
#以下为自定义评分，采用bert
#自定义评分开始
import numpy as np
from numpy.linalg import norm
try:
    import sentence_transformers
except ImportError:
    os.system("pip install sentence-transformers")
    import sentence_transformers
def evaluator(question:str,res:list):
    model=sentence_transformers.SentenceTransformer('bert-base-nli-mean-tokens')#国内连不上
    scores=[]
    for dic in res:
        embeddings=model.encode([question,dic['text']])
        similarity=np.dot(embeddings[0],embeddings[1])/(norm(embeddings[0])*norm(embeddings[1]))
        scores.append(float(similarity))
    return scores
#自定义评分结束


#主功能函数
def LOAD_DOC_ASK_LLM_AND_RETRIEVE(
    question,
    document_path,
    storage_path=None,
    similarity_top_k=5,
    similarity_cutoff=0.6,
    ):
    #设置开始
    os.environ["REPLICATE_API_TOKEN"] = "r8_egNIICyoySVOijfalSoobPDs2llj0pb4XTznG"

    # set the LLM
    llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
    Settings.llm = Replicate(
        model=llama2_7b_chat,
        temperature=0.02,
        additional_kwargs={"top_p": 1, "max_new_tokens": 800},
    )

    # set tokenizer to match LLM
    Settings.tokenizer = AutoTokenizer.from_pretrained(
        "NousResearch/Llama-2-7b-chat-hf"#国内连不上
    )

    # set the embed model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-zh-v1.5"#国内连不上
    )
    #设置结束

    #text splitter设置开始

    splitter = SentenceSplitter(chunk_size=256,chunk_overlap=128)

    #text splitter设置结束


    #加载文件
    documents = SimpleDirectoryReader(document_path).load_data()
    #分割，转为nodes
    nodes = splitter.get_nodes_from_documents(documents)
    #转为index和vector
    index = VectorStoreIndex(nodes)
    #检索器设置
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,  #设置这个值，可以改变符合要求条数的上限
    )

    #设置重排器
    colbert_reranker = ColbertRerank(
        top_n=5,
        model="colbert-ir/colbertv2.0",
        tokenizer="colbert-ir/colbertv2.0",
        keep_retrieval_score=True,
    )

    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=None,  #如果是None，自动采用设置的llm判断相似度
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff), colbert_reranker],#这里相似度要求设置为0.6以上，可以改#添加重排器
    )

    #提问
    question=question
    response=query_engine.query(question)
    #显示回答
    print(question)
    print(response)

    #显示相关文档
    res=[]
    for node in response.source_nodes:
        if node.get_score()!=None:
            res.append({'text':node.text,'colbert_score':node.score})
    

    bert_scores=evaluator(question=question,res=res)
    for i,dic in enumerate(res):
        dic['bert_score']=bert_scores[i]
    #print("基于BERT的评分)\n\n",bert_scores)
    bert_score = sorted(res, key=lambda dic: dic['bert_score'], reverse=True)
    print("\n基于BERT的评分排序:\n", bert_score)

    # 基于Colbert重排器的评分排序
    print("\n基于Colbert重排后的自评分排序:\n", res)

    
    
#    print("\n",res)
    #将向量索引库存入本地
    if storage_path!=None:
        index.storage_context.persist(storage_path)#将向量和索引导入到本地
    
    average_score=sum(bert_scores)/len(bert_scores)
    # 计算 colbert_scores 的平均值
    colbert_scores = [dic['colbert_score'] for dic in res]
    average_colbert_score = sum(colbert_scores) / len(colbert_scores)
    json_res=json.dumps(res)
    return json_res,average_score,average_colbert_score#返回每个相似度符合要求的文本，以及其自带评分，为json文件

#调用示例
res,bert_score,colbert_score=LOAD_DOC_ASK_LLM_AND_RETRIEVE(question="核型多角体病毒是什么？有什么作用和危害？",document_path='data',storage_path='storage')
