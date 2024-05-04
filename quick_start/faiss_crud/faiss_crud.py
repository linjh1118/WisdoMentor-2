import fire
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import utils.store.faiss.crud_client as faiss_client_py
import numpy as np


faiss_client = faiss_client_py.CrudClient("http://localhost:8889")
    
# 1. how to search
print('===== 1 =====')
## search from text
query, top_k ="Knowledge Distillation", 3
res = faiss_client.search_vec_from_text(query = query, top_k=top_k)
print('indices:', res['indices'])
print('knowledges:', res['knowledges'])
## search from vec
res = faiss_client.search(query_vec = np.arange(1024).tolist(), top_k=top_k)  
print('indices:', res['indices'])
print('knowledges:', res['knowledges'])

# 2. get
print('===== 2 =====')
vector = faiss_client.get_vector(4)
chunk = faiss_client.get_chunk(4)
print('vector:', vector[:10])
print('chunk:', chunk)

"""  暂时放弃
# 3. update. 将5号改成4号嵌入，之前召回的是[4, 11, 3]。 若是成功更新，则应该召回[4, 5, 11]

"""

print('===== 3 =====')
res = faiss_client.search_vec_from_text(query = query, top_k=top_k)
print('indices:', res['indices'])
# faiss_client.update_vector(index_id=5, new_vec=faiss_client.get_vector(4))
faiss_client.update_vector(index_id=5, new_vec=np.arange(1024).tolist())
print('vector4:', faiss_client.get_vector(4)[:10])
print('vector5:', faiss_client.get_vector(5)[:10])
res = faiss_client.search_vec_from_text(query = query, top_k=top_k)
print('indices:', res['indices'])

# 4. add
faiss_client.add_vector(index_id=21, new_vec=(np.arange(1024) + np.arange(1024)).tolist())
print('vector21:', faiss_client.get_vector(21)[:10])