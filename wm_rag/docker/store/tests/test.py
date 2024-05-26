import requests

# print(index.get_id_by_doc("apple"))

# id = index.get_id_by_doc("apple")
# index.delete_by_id(id)

requests.delete(url="http://localhost:10001/ann/remove_items", json={"ids": [-1]})

sentences = ["apple", "banana"]

embs = requests.post(
    url="http://localhost:10000/get_embedding/",
    json={"text": sentences, "model_type": "bert"},
).json()["embedding"]

requests.post(
    url="http://localhost:10001/ann/add_docs",
    json={"doc_list": sentences, "doc_emb_list": embs},
)

requests.post(
    url="http://localhost:10001/ann/add_docs",
    json={"doc_list": sentences, "doc_emb_list": embs},
)

sentences = ["pear", "watermelon"]

embs = requests.post(
    url="http://localhost:10000/get_embedding/",
    json={"text": sentences, "model_type": "bert"},
).json()["embedding"]

requests.post(
    url="http://localhost:10001/ann/add_docs",
    json={"doc_list": sentences, "doc_emb_list": embs},
)

search_sentences = ["app"]
search_embs = requests.post(
    url="http://localhost:10000/get_embedding/",
    json={"text": search_sentences, "model_type": "bert"},
).json()["embedding"]

search_res = requests.post(
    url="http://localhost:10001/ann/search", json={"query_vec": search_embs, "num": 2}
).json()["knowledges"]
print("search res", search_res)

ids = requests.post(
    url="http://localhost:10001/ann/get_ids", json={"docs": ["apple", "banana"]}
).json()["ids"]

print("ids", ids)

requests.delete(url="http://localhost:10001/ann/remove_items", json={"ids": ids})

new_ids = requests.post(
    url="http://localhost:10001/ann/get_ids", json={"docs": ["apple", "banana"]}
).json()["ids"]

print("ids", new_ids)
# index.add_documents(sentences, embs)

# emb = requests.post(
#             url="http://localhost:10000/get_embedding/",
#             json={
#                 "text": ["app"],
#                 "model_type": "bert"
#             }
#         ).json()["embedding"][0]

# res = index.search_by_embedding([emb])
# for r in res:
#     print(r)

# sentences = [
#     "test1",
#     "test2"
# ]

# embs = requests.post(
#             url="http://localhost:10000/get_embedding/",
#             json={
#                 "text": sentences,
#                 "model_type": "bert"
#             }
#         ).json()["embedding"]

# index.add_documents(sentences, embs)

# index.add_documents(sentences, embs)

# new_sentences = [
#     "test3",
#     "test4",
#     "test5"
# ]

# new_embs = requests.post(
#             url="http://localhost:10000/get_embedding/",
#             json={
#                 "text": new_sentences,
#                 "model_type": "bert"
#             }
#         ).json()["embedding"]

# index.add_documents(new_sentences, new_embs)
