from typing import List
from abc import abstractmethod
from base import Query

from llama_index.core import Settings,VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.indices.query.query_transform import HyDEQueryTransform
from llama_index.core.query_engine import TransformQueryEngine

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.replicate import Replicate
from transformers import AutoTokenizer

import os
os.environ["REPLICATE_API_TOKEN"] = "r8_PXRtGJ1M2LR0Y6fEqL4PcHfEnOuOiOk3NgqiJ"

# set the LLM
llama2_7b_chat = "meta/llama-2-7b-chat:8e6975e5ed6174911a6ff3d60540dfd4844201974602551e10e9e87ab143d81e"
Settings.llm = Replicate(
    model=llama2_7b_chat,
    temperature=0.01,
    additional_kwargs={"top_p": 1, "max_new_tokens": 300},
)
# set tokenizer to match LLM
Settings.tokenizer = AutoTokenizer.from_pretrained(
    "NousResearch/Llama-2-7b-chat-hf"
)
# set the embed model
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-zh-v1.5"
)


class MyQuery(Query):
    def __init__(self) -> None:
        super().__init__()
        

    
    def query_extension(self, query: str) -> List[str]:
        pass

    
    def query_rewrite(self, query: str) -> List[str]:
        original_query=query
        
        # Load documents, build the VectorStoreIndex
        dir_path = "C:\\Users\\HP\\Desktop\\llama index索引\\data"
        documents = SimpleDirectoryReader(dir_path).load_data()
        index = VectorStoreIndex.from_documents(documents)
        
        query_engine = index.as_query_engine()

        # Query with HyDE transformation
        hyde = HyDEQueryTransform(include_original=True)
        hyde_query_engine = TransformQueryEngine(query_engine, hyde)
        response = hyde_query_engine.query(original_query)
        return response




shili=MyQuery()
res=shili.query_rewrite("What makes nuclear polyhedrosis viruses special?")
print(res)
