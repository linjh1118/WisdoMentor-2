import sys

sys.path.append(f"{sys.path[0]}/..")

from langchain_core.documents import Document

from store.ann_store import AnnStore
from embedding.bge_embedding import BgeEmbedding

store = AnnStore(router_path="http://localhost", port="10001")
embedding = BgeEmbedding(route_path="http://localhost", port="10000")

store_doc = [Document(page_content="apple"), Document(page_content="banana")]
store_embs = embedding.embed_documents(store_doc)

store.add_documents(store_doc, store_embs)

query_doc = [Document(page_content="app")]
query_embs = embedding.embed_documents(query_doc)

res = store.search_by_embed(query_embs, k=10)

for r in res:
    print(r.page_content)

ids = store.get_id_by_docs(store_doc)
store.delete_documents_by_ids(ids)

res = store.search_by_embed(query_embs, k=10)

for r in res:
    print(r.page_content)
