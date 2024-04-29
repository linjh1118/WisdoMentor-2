import nest_asyncio
import logging
import sys
import os
import requests
from pathlib import Path
import pandas as pd
from IPython.display import display, HTML
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings, QueryBundle
from llama_index.core.postprocessor import LLMRerank
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank
from llama_index.core.retrievers import VectorIndexRetriever

# Apply nest_asyncio
nest_asyncio.apply()

# Set up logging
logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logging.getLogger().addHandler(logging.StreamHandler(stream=sys.stdout))

# Set up LLM and embedding model
llm = Ollama(model="gemma:2b", temperature=0.1)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="D:/llama-index/llama_index/Rerank/bge-base-en-v1.5")
Settings.chunk_size = 512

# Set up data path
data_path = Path("data_wiki")

# Set up wiki titles
wiki_titles = ["Vincent van Gogh"]

# Fetch and save wiki data
for title in wiki_titles:
    response = requests.get(
        "https://en.wikipedia.org/w/api.php",
        params={
            "action": "query",
            "format": "json",
            "titles": title,
            "prop": "extracts",
            "explaintext": True,
        },
    ).json()
    page = next(iter(response["query"]["pages"].values()))
    wiki_text = page["extract"]

    if not data_path.exists():
        Path.mkdir(data_path)

    with open('your_file_path', 'w', encoding='utf-8') as fp:
        fp.write(wiki_text)

# Load documents
documents = SimpleDirectoryReader("./data_wiki/").load_data()

# Create index
index = VectorStoreIndex.from_documents(documents)

# Define function to get retrieved nodes
def get_retrieved_nodes(query_str, vector_top_k=5, reranker_top_n=3, with_reranker=False):
    query_bundle = QueryBundle(query_str)
    retriever = VectorIndexRetriever(index=index, similarity_top_k=vector_top_k)
    retrieved_nodes = retriever.retrieve(query_bundle)

    if with_reranker:
        reranker = RankGPTRerank(llm=llm, top_n=reranker_top_n, verbose=True)
        retrieved_nodes = reranker.postprocess_nodes(retrieved_nodes, query_bundle)

    return retrieved_nodes

# Define function to pretty print dataframe
def pretty_print(df):
    return display(HTML(df.to_html().replace("\\n", "")))

# Define function to visualize retrieved nodes
def visualize_retrieved_nodes(nodes):
    result_dicts = []
    for node in nodes:
        result_dict = {"Score": node.score, "Text": node.node.get_text()}
        result_dicts.append(result_dict)

    pretty_print(pd.DataFrame(result_dicts))

# Retrieve and rerank nodes
new_nodes = get_retrieved_nodes(
    "Which date did Paul Gauguin arrive in Arles ?",
    vector_top_k=10,
    reranker_top_n=3,
    with_reranker=True,
)