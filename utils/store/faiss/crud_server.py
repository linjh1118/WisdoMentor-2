from typing import List
from fastapi import FastAPI, HTTPException # 0.109.0
from pydantic import BaseModel 
import faiss # '1.7.4'
from contextlib import asynccontextmanager
import argparse
import sys
import os
import json
import numpy as np
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
from base import *
import utils.embed_model.bge.bge_embed_client as bge_client
client = bge_client.BGEClient("http://localhost:8888/get_embedding/")


"""
reference: 
1. https://fastapi.tiangolo.com/advanced/events/#alternative-events-deprecated
2. https://github.com/facebookresearch/faiss/wiki/Getting-started
"""

# === load config ===
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index_file', type=str, default='/home/chy/dream/linjh/rag/wisdomentor/utils/store/faiss/hello_data/database.index')
    parser.add_argument('--port', type=int, default=8889)
    args = parser.parse_args()
    return args

args = get_args()
logger = set_naive_logger('../../../log_cache')
log_hyperparams(args)
index = None # for global
index2chunk = None
index_file = args.index_file


# prepare data class
class Vector(BaseModel):
    vector: list[float]
    
class QueryAndTopk(BaseModel):
    query: str
    top_k: int

class QueryVecAndTopk(BaseModel):
    query_vec: list[float]
    top_k: int

class IndexId(BaseModel):
    index_id: int
    
class IndexIdAndNewVec(BaseModel):
    index_id: int
    new_vec: list[float]


# === define startup and endup (load and save index and so on) ===
def load_index_from_file():
    global index, index2chunk
    index = faiss.read_index(index_file)
    index2chunk = json.load(open(index_file + '2chunk', 'r'))
    # logger.info(f'index2chunk: {index2chunk}')
    

def save_index_to_file():
    faiss.write_index(index, index_file + '_new')

@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    logger.info('Loading index from file')
    load_index_from_file()
    yield
    # shutdown
    logger.info("Save index to file")
    save_index_to_file()

app = FastAPI(lifespan=lifespan)


# === define CRUD function ===
@app.post('/faiss/search_from_text')  # notice: don't type "/faiss/search_vec_from_text/" in curl
def search_from_text(queryAndTopk: QueryAndTopk):
    query, top_k = queryAndTopk.query, queryAndTopk.top_k
    logger.info(f'query: {query}'); logger.info(f'top_k: {top_k}')
    embedding = client.get_embedding(text = query)
    _, indices = index.search(np.array([embedding]), top_k)
    indices = indices[0].tolist()
    knowledges = [index2chunk[str(idx)] for idx in indices]
    return {'indices': indices, 'knowledges': knowledges}

@app.post('/faiss/search')
def search(queryVecAndTopk: QueryVecAndTopk):
    query_vec, top_k = queryVecAndTopk.query_vec, queryVecAndTopk.top_k
    assert len(query_vec) == index.d, f"查询向量的维度与向量库中的向量维度不匹配, 二者分别为: {len(query_vec)}, {index.d}"
    logger.info(f'top_k: {top_k}')
    _, indices = index.search(np.array([query_vec]), top_k)
    indices = indices[0].tolist()
    knowledges = [index2chunk[str(idx)] for idx in indices]
    return {'indices': indices, 'knowledges': knowledges}

@app.post('/faiss/get_vector')
def get_vector(index_id: IndexId):
    # global index
    index_id = index_id.index_id
    if index_id < 0 or index_id >= index.ntotal:
        raise HTTPException(status_code=404, detail="Vector not found")
    vector = index.reconstruct(index_id)
    return {"vector": vector.tolist()}

@app.post('/faiss/get_chunk')
def get_chunk(index_id: IndexId):
    # global index
    index_id = index_id.index_id
    if index_id < 0 or index_id >= index.ntotal:
        raise HTTPException(status_code=404, detail="Vector not found")
    chunk = index2chunk[str(index_id)]
    return {"chunk": chunk}

@app.post('/faiss/update_vector')
def update_vector(indexIdAndNewVec: IndexIdAndNewVec):
    global index
    index_id, new_vec = indexIdAndNewVec.index_id, indexIdAndNewVec.new_vec
    if index_id < 0 or index_id >= index.ntotal:
        raise HTTPException(status_code=404, detail="Vector not found")
    vector_tensor = np.array(new_vec, dtype=np.float32)  # 错误竟然出在这里！add是默认，更新默认就不行了。。
    logging.info(f'{vector_tensor.dtype}')
    logger.info(f'start update vector {index_id}')
    logger.info(f'vec before: {index.reconstruct(index_id)[:10]}')
    index.reconstruct(index_id, vector_tensor)
    # index.replace(3, vector_tensor)
    # index.update_vectors(faiss.IndexedVectors([new_vec]), [3])
    logger.info(f'vec after: {index.reconstruct(index_id)[:10]}')
    logger.info(f'success update vector {index_id}')
    return {"message": f"Vector {index_id} updated successfully"}

@app.delete('/faiss/delete')
def delete_vector(index_id: IndexId):
    """https://blog.csdn.net/u013066730/article/details/106298939"""
    # global index
    index_id = index_id.index_id
    if index_id < 0 or index_id >= index.ntotal:
        raise HTTPException(status_code=404, detail="Vector not found")
    index.remove_ids(faiss.IDSelectorArray([index_id]))
    return {"message": "Vector deleted successfully"}

# ===这个函数需要对一下，其他应该没有问题了==== 
@app.post('/faiss/add_vector')
def add_vector(indexIdAndNewVec: IndexIdAndNewVec):
    global index
    index_id, new_vec = indexIdAndNewVec.index_id, indexIdAndNewVec.new_vec
    vector_tensor = np.array([new_vec], dtype=np.float32)
    # 1. 
    # dexIDMap.cpp:32: Error: 'index->ntotal == 0' failed: index must be empty on input
    # index2 = faiss.IndexIDMap(index)
    # index2.add_with_ids(vector_tensor, [index.ntotal + 1])
    # 2. 
    logging.info(f'index.ntotal: {index.ntotal}')
    index.add(vector_tensor)  # 不知道有没有加成功。
    logging.info(f'index.ntotal: {index.ntotal}')
    
    return {"message": "Vector added successfully"}
    """
    https://www.volcengine.com/theme/4960319-R-7-1
    https://cloud.tencent.com/developer/ask/sof/106820413
    import numpy as np import faiss # 准备原始数据和对应的索引文件：train_data.npy和train_index.faissindex # 读取数据 data = np.load('train_data.npy') index = faiss.read_index('train_index.faissindex') # 创建IncrementalIndex实例 incremental_index = faiss.IncrementalIndex(index.ntotal, index.d) # 添加新的数据 new_data = np.random.rand(100, 128).astype('float32') new_data_ids = np.arange(index.ntotal, index.ntotal + len(new_data)) # 每条数据分配一个唯一ID incremental_index.add_with_ids(new_data, new_data_ids) # 重新训练索引 incremental_index.train() incremental_index.add(incremental_index.swig_ptr()) # 添加索引到原有索引矩阵中 # 保存索引 faiss.write_index(incremental_index, 'incremental_index.faissindex')
    """


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)

# 启动指令
# pyrag crud_server.py
