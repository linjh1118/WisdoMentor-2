import os
import requests
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.jinaai import JinaEmbedding
from llama_index.postprocessor.jinaai_rerank import JinaRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings
from llama_index.llms.ollama import Ollama
# 初始化模型
llm = Ollama(model="gemma:2b")
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="bge-base-en-v1.5")
# Set up JinaEmbedding
api_key = "jina_f4dd227a001140f0b7f9f2d815a49a53PeuER-IsToA3A1VxBlVnqMvp9lo5"
jina_embeddings = JinaEmbedding(api_key=api_key)

# Download PDF file
url = "https://niketeam-asset-download.nike.net/catalogs/2024/2024_Nike%20Kids_02_09_24.pdf?cb=09302022"
response = requests.get(url)
with open("Nike_Catalog.pdf", "wb") as f:
    f.write(response.content)

# Load documents from PDF file
reader = SimpleDirectoryReader(input_files=["Nike_Catalog.pdf"])
documents = reader.load_data()

# Build index from documents
index = VectorStoreIndex.from_documents(documents=documents, embed_model=jina_embeddings)

# Set up JinaRerank
jina_rerank = JinaRerank(api_key=api_key, top_n=2)

# Create query engine
query_engine = index.as_query_engine(similarity_top_k=10, node_postprocessors=[jina_rerank])

# Perform query and print results
response = query_engine.query("What is the best jersey by Nike in terms of fabric?")
print(response.source_nodes[0].text, response.source_nodes[0].score)
print("\n")
print(response.source_nodes[1].text, response.source_nodes[1].score)