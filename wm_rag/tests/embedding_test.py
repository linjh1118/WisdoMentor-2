from langchain_core.documents import Document
import sys

sys.path.append(f"{sys.path[0]}/..")

from embedding.bert_embedding import BertEmbedding
from embedding.bge_embedding import BgeEmbedding

text = "微生物是什么？"

documents = [Document(page_content=text)]

bert = BertEmbedding(route_path="http://localhost", port="10000")
bge = BgeEmbedding(route_path="http://localhost", port="10000")

bert_text_res = bert.embed_text(text)
bge_text_res = bge.embed_text(text)

print("bert len: ", len(bert_text_res))
print("bge len: ", len(bge_text_res))

bert_documents_res = bert.embed_documents(documents)
bge_documents_res = bge.embed_documents(documents)

print("bert_text_res:", bert_text_res[:3])
print("bge_text_res:", bge_text_res[:3])
print("bert_documents_res:", bert_documents_res[0][:3])
print("bge_documents_res:", bge_documents_res[0][:3])
