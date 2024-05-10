# faiss服务调用指南

### 1. 启动服务
``` bash
$ pyrag crud_server.py 
输出：
2024-05-04 18:50:33,264 INFO     index_file = /home/chy/dream/linjh/rag/wisdomentor/utils/store/faiss/hello_data/database.index
2024-05-04 18:50:33,264 INFO     port = 8889
INFO:     Started server process [1399293]
INFO:     Waiting for application startup.
2024-05-04 18:50:33,305 INFO     Loading index from file
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8889 (Press CTRL+C to quit)

# 服务异常，可以通过下面指令检查
query="Knowledge Distillation" && curl -X POST -H "Content-Type: application/json" -d "{\"query\": \"${query}\", \"top_k\": 3}" http://localhost:8889/faiss/search_from_text > ../../log_cache/log_faiss_crud_select
```
    
### 2. 调用服务

1. 通过`CrudClient` 调用 
``` python
import utils.store.faiss.crud_client as faiss_client_py
faiss_client = faiss_client_py.CrudClient("http://localhost:8889")
query, top_k = 'Knowledge Distillation', 3
res = faiss_client.search_vec_from_text(query = query, top_k=top_k)
print('indices:', res['indices'])
print('knowledges:', res['knowledges'])

结果如下所示：
indices: [4, 11, 3]
knowledges: ['Investigating the Impact of Transfer Learning and Knowledge Distillation on Small Data Classification Tasks', 'A Hybrid Approach for Intelligent Tutoring Systems: Integrating Knowledge Tracing with Reinforcement Learning', 'Enhancing Natural Language Understanding in Chatbot Systems using Transformer-based Models and Knowledge Graphs']
```

2. 上面是调用了`search_vec_from_text`接口，这里将`crud_client`中的所有接口都展示一下

``` python
def search_vec_from_text(self, query: str, top_k: int): ->
Dict {
    "indices": list[int],
    "knowledges": list[str]
}



def search_from_text(self, query: str, top_k: int): ->
Dict {
    "indices": list[int],
    "knowledges": list[str]
}

def get_chunk(self, index_id: int): -> chunk: str

def get_vector(self, index_id: int): -> vector: list[float]

def remove(self, index_id: int) -> None

# 以下两个有问题，正在修复。目前的一种新想法就是，显式保存一个向量文件，然后每次就新建一个faiss
def update_vector(self, index_id: int, new_vec: list[float]):
def add_vector(self, index_id: int, new_vec: list[float]):
```