from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
import os

#OPENAI_API_TOKEN = "sk-"
#os.environ["OPENAI_API_KEY"] = OPENAI_API_TOKEN
# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
#Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)
llm = Ollama(model="gemma:2b")
Settings.llm = llm
#Settings.embed_model = HuggingFaceEmbedding(model_name="D:/llama-index/llama_index/Rerank/bge-base-en-v1.5")

# build index
index = VectorStoreIndex.from_documents(documents=documents)
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)

rerank = FlagEmbeddingReranker(model="BAAI/bge-reranker-large", top_n=5)
from time import time
query_engine = index.as_query_engine(
    similarity_top_k=10, node_postprocessors=[rerank]
)

now = time()
response = query_engine.query(
    "Which grad schools did the author apply for and why?",
)
print(f"Elapsed: {round(time() - now, 2)}s")
print(response)
print(response.get_formatted_sources(length=200))
