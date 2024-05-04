import requests

class CrudClient:
    def __init__(self, url):
        self.url = url

    def search_vec_from_text(self, query: str, top_k: int):
        data = {"query": query, "top_k": top_k}
        postfix = '/faiss/search_from_text'
        response = requests.post(self.url + postfix, json=data)
        return response.json()
    
    def search(self, query_vec: list[float], top_k: int):
        data = {"query_vec": query_vec, "top_k": top_k}
        postfix = '/faiss/search'
        response = requests.post(self.url + postfix, json=data)
        return response.json()
    
    def get_chunk(self, index_id: int):
        data = {"index_id": index_id}
        postfix = '/faiss/get_chunk'
        response = requests.post(self.url + postfix, json=data)
        return response.json()['chunk']
    
    def get_vector(self, index_id: int):
        data = {"index_id": index_id}
        postfix = '/faiss/get_vector'
        response = requests.post(self.url + postfix, json=data)
        return response.json()['vector']
    
    # 上边已全部支持。下边的仍存问题。
    
    def update_vector(self, index_id: int, new_vec: list[float]):
        data = {"index_id": index_id, "new_vec": new_vec}
        postfix = '/faiss/update_vector'
        response = requests.post(self.url + postfix, json=data)
        return response.json()['message']
    
    def add_vector(self, index_id: int, new_vec: list[float]):
        data = {"index_id": index_id, "new_vec": new_vec}
        postfix = '/faiss/add_vector'
        response = requests.post(self.url + postfix, json=data)
        return response.json()['message']
    
    def remove_vector(self, item_idx_list: int):
        data = {"item_idx_list": item_idx_list}
        postfix = '/faiss/remove_vector'
        response = requests.post(self.url + postfix, json=data)
        return response.json()['message']
    
""" How to use
拿一个函数举例子search_from_text，可以用python或者curl调用

1. python: 调用CrudClient
``` python
import utils.store.faiss.crud_client as faiss_client_py
faiss_client = faiss_client_py.CrudClient("http://localhost:8889")
query, top_k = 'Knowledge Distillation', 3
res = faiss_client.search_vec_from_text(query = query, top_k=top_k)
print('indices:', res['indices'])
print('knowledges:', res['knowledges'])
# 结果如下所示：
# indices: [4, 11, 3]
# knowledges: ['Investigating the Impact of Transfer Learning and Knowledge Distillation on Small Data Classification Tasks', 'A Hybrid Approach for Intelligent Tutoring Systems: Integrating Knowledge Tracing with Reinforcement Learning', 'Enhancing Natural Language Understanding in Chatbot Systems using Transformer-based Models and Knowledge Graphs']
```

2. curl
``` bash
query="Knowledge Distillation" && curl -X POST -H "Content-Type: application/json" -d "{\"query\": \"${query}\", \"top_k\": 3}" http://localhost:8889/faiss/search_from_text > ../../log_cache/log_faiss_crud_select
输出的结果如下所示：
{
    "indices": [
        4,
        11,
        3
    ],
    "knowledges": [
        "Investigating the Impact of Transfer Learning and Knowledge Distillation on Small Data Classification Tasks",
        "A Hybrid Approach for Intelligent Tutoring Systems: Integrating Knowledge Tracing with Reinforcement Learning",
        "Enhancing Natural Language Understanding in Chatbot Systems using Transformer-based Models and Knowledge Graphs"
    ]
}
```



"""
