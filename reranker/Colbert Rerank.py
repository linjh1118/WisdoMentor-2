from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
)
import os
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
llm = Ollama(model="gemma:2b", temperature=0.1)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="D:/llama-index/llama_index/Rerank/bge-base-en-v1.5")

#os.environ["OPENAI_API_KEY"] = "sk-"
# load documents
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# build index
index = VectorStoreIndex.from_documents(documents=documents)


from llama_index.postprocessor.colbert_rerank import ColbertRerank

colbert_reranker = ColbertRerank(
    top_n=5,
    model="colbert-ir/colbertv2.0",
    tokenizer="colbert-ir/colbertv2.0",
    keep_retrieval_score=True,
)

query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[colbert_reranker],
)
response = query_engine.query(
    "What did Sam Altman do in this essay?",
)
for node in response.source_nodes:
    print(node.id_)
    print(node.node.get_content()[:120])
    print("reranking score: ", node.score)
    print("retrieval score: ", node.node.metadata["retrieval_score"])
    print("**********")

print(response)

response = query_engine.query(
    "Which schools did Paul attend?",
)
for node in response.source_nodes:
    print(node.id_)
    print(node.node.get_content()[:120])
    print("reranking score: ", node.score)
    print("retrieval score: ", node.node.metadata["retrieval_score"])
    print("**********")

print(response)