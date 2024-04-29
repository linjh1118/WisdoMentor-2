import os
import json


from llama_index.core import StorageContext, load_index_from_storage
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
from transformers import AutoTokenizer


from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.indices.postprocessor import SimilarityPostprocessor

#以下为自定义评分，采用bert评分方法
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
def LOAD_DISK_ASK_LLM_AND_RETRIEVE(
    question,
    storage_path="./storage",
    similarity_top_k=5,
    similarity_cutoff=0.6,
    ):
    #设置开始
    os.environ["REPLICATE_API_TOKEN"] = "r8_9kb8AfmUIswwZdoTYgyqrSbWehJ5f1y4SqN13"

    # set the LLM
    llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
    Settings.llm = Replicate(
        model=llama2_7b_chat,
        temperature=0.01,
        additional_kwargs={"top_p": 1, "max_new_tokens": 300},
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


    # rebuild storage context
    storage_context = StorageContext.from_defaults(persist_dir=storage_path)
    # load index
    index = load_index_from_storage(storage_context)
    #检索器设置
    retriever = VectorIndexRetriever(
        index=index,
        similarity_top_k=similarity_top_k,  #设置这个值，可以改变符合要求条数的上限
    )


    query_engine = RetrieverQueryEngine(
        retriever=retriever,
        response_synthesizer=None,  #如果是None，自动采用设置的llm判断相似度
        node_postprocessors=[SimilarityPostprocessor(similarity_cutoff=similarity_cutoff)],#这里相似度要求设置为0.6以上，可以改
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
            res.append({'text':node.text,'self_score':node.score})
    

    bert_scores=evaluator(question=question,res=res)
    print("基于BERT的评分\n\n",bert_scores)
    for i,dic in enumerate(res):
        dic['bert_score']=bert_scores[i]
    
    print("\n",res)
    
    average_score=sum(bert_scores)/len(bert_scores)
    json_res=json.dumps(res)
    return json_res,average_score#返回每个相似度符合要求的文本，以及其自带评分，为json文件

#调用示例
res,bert_score=LOAD_DISK_ASK_LLM_AND_RETRIEVE(question="核型多角体病毒有什么特点和危害？",storage_path='storage')