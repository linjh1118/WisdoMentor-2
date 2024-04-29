from llama_index.core import VectorStoreIndex, SimpleDirectoryReader

# load documents
documents = SimpleDirectoryReader("./data/paul_graham").load_data()

from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
#from llama_index.llms.openai import OpenAI
from llama_index.llms.ollama import Ollama
# 初始化模型
Settings.llm = Ollama(model="gemma:2b")
#Settings.llm = OpenAI(model="gpt-3.5-turbo")
Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-small-en-v1.5"
)

# build index
index = VectorStoreIndex.from_documents(documents=documents)

from llama_index.core.postprocessor import SentenceTransformerRerank

rerank = SentenceTransformerRerank(
    model="cross-encoder/ms-marco-MiniLM-L-2-v2", top_n=3
)

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

