"""
接口设计方案:
### 召回
1. search_from_text(query: str)
普通的查
2. search_from_idx(idx: int)
检索出 索引号对应的物体最近的物体。用处：query从各个方向找到一个物体，然后再根据召回物体的索引号找到最近的物体（ie，二跳）。
3. item
三者暂时用一个后端函数统一了，但是client中依然分成了三个

### 改
3. add接口（往起加向量）
4. update接口（更新向量）
5. remove接口（删除某idx）
6. saveAndBuild接口（add和update之后，要重新建库，saveandbuild）
7. load接口（加载某一个ann库）

### 查
8. show_ids
9. get_vector
"""


import os
import json
from annoy import AnnoyIndex
import argparse
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import embed_model.bge.bge_embed_client as bge_client
client = bge_client.BGEClient("http://localhost:8888/get_embedding/")
from base import *
from store import data_class
from collections import defaultdict
from tqdm import tqdm
from typing import List
from fastapi import FastAPI, HTTPException # 0.109.0

import faiss # '1.7.4'
from contextlib import asynccontextmanager

"""
reference:
https://github.com/spotify/annoy/issues/411
Q: Add more items to an index which is loaded from disk #411
A: This is unfortunately not possible

Linjh: So, all i can do is unload, then add, finally save in an overwrite way.
"""

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--database_dir', type=str, default='hello_data')
    parser.add_argument('--database_id', type=str, default='database')
    parser.add_argument("--dim", default=1024, type=int, help="知识块嵌入维度")
    parser.add_argument("--tree_size", default=100, type=int, help="tree_size in ann")
    parser.add_argument('--port', type=int, default=8890)
    args = parser.parse_args()
    return args

args = get_args()
logger = set_naive_logger('../../../log_cache', log_name='ann_server')
log_hyperparams(args)

ann_binary_file = os.path.join(args.database_dir, args.database_id + '.ann')
idx_map_file = os.path.join(args.database_dir, args.database_id + '.idx2fea.jsonl')

idx2feature = {}
ann = AnnoyIndex(args.dim, 'angular')

def load_ann_from_disk(idx_map_file=idx_map_file, ann_binary_file=ann_binary_file):
    logger.info(f'Load Ann from {ann_binary_file}, {idx_map_file}')
    for l in get_lines(idx_map_file):
        cur_dict = json.loads(l)
        idx2feature[cur_dict['idx']] = cur_dict['fea']
    ann.load(ann_binary_file, prefault=True)
    logger.info('Done loading')
    
def save_ann_to_disk(idx_map_file=idx_map_file, ann_binary_file=ann_binary_file):
    logger.info(f'Notice! Overwrite Ann to init file {ann_binary_file}, {idx_map_file}')
    ann.save(args.ann_binary_file)
    with open(args.ann_binary_file.replace(".ann", ".idx2fea.jsonl"), 'w') as f:
        for idx, fea in idx2feature.items():
            f.write(json.dumps({'idx': idx, 'fea': fea}) + '\n')
    logger.info('Done saving')


@asynccontextmanager
async def lifespan(app: FastAPI):
    # startup
    load_ann_from_disk()
    yield
    # shutdown
    save_ann_to_disk()

app = FastAPI(lifespan=lifespan)


# === define backend function ===

# 通过设定pydantic.BaseModel的None值
@app.post('/ann/search')
def search(queryAndTopkANN: data_class.QueryAndTopkANN):
    query, top_k, beam_k = queryAndTopkANN.query, queryAndTopkANN.top_k, queryAndTopkANN.beam_k
    logger.info(f'query: {query}'); logger.info(f'top_k: {top_k}'); logger.info(f'beam_k: {beam_k}')
    if queryAndTopkANN.item_idx is None:
        if queryAndTopkANN.query_vec is None:
            query_emb = client.get_embedding(text = query)
        else:
            query_emb = queryAndTopkANN.query_vec
        indices, distances = ann.get_nns_by_vector(query_emb, n=top_k, include_distances=True, search_k=beam_k)
    else:
        indices, distances = ann.get_nns_by_item(queryAndTopkANN.item_idx, n=top_k, include_distances=True, search_k=beam_k)
    knowledges = [idx2feature[idx]['text'] for idx in indices]
    return {'indices': indices, 'knowledges': knowledges}

loaded = True
change_history = []
@app.post('/ann/show_all_idx')
def show_all_idx():
    return {'all_idx': list(idx2feature.keys())}

@app.post('/ann/get_emb_of_item')
def get_emb_of_item(itemIdx: data_class.ItemIdx):
    return {'emb': idx2feature[itemIdx.item_idx]['bge_emb']}


@app.post('/ann/add_docs')
def add_docs(docListAndEmb: data_class.DocListAndEmb):
    """Note: please call build_and_save after you finish your several add calling"""
    assert docListAndEmb.doc_list is not None or docListAndEmb.doc_emb_list is not None
    global ann, idx2feature, loaded, change_history
    if loaded:
        # first time add
        loaded = False
        logger.info(f'At begin, ann.get_n_items(): {ann.get_n_items()}')
        ann.unload()
        logger.info(f'After unloading, ann.get_n_items(): {ann.get_n_items()}')
        ann = AnnoyIndex(args.dim, 'angular')
        
    # 修改idx2feature，然后重新建库，最后覆盖保存. 后两部解耦出去。
    ### 0. complete embedding if need
    doc_list = docListAndEmb.doc_list
    emb_list = docListAndEmb.doc_emb_list
    if emb_list is None:
        emb_list = [client.get_embedding(text) for text in tqdm(doc_list, desc=f'generating embedding')]
    ### 1. modify idx2feature
    start_idx = max(idx2feature.keys()) + 1
    for idx, (text, emb) in enumerate(zip(doc_list, emb_list), start=start_idx):
        idx2feature[idx] = {'text': text, 'bge_emb': emb}
        change_history.append(f'add text: {text}')
    

@app.post('/ann/remove_items')
def remove_items(removeOrUpdateItemList: data_class.RemoveOrUpdateItemList):
    """Note: please call build_and_save after you finish your several changing calling"""
    global ann, idx2feature, loaded, change_history
    if loaded:
        # first time add
        loaded = False
        logger.info(f'At begin, ann.get_n_items(): {ann.get_n_items()}')
        ann.unload()
        logger.info(f'After unloading, ann.get_n_items(): {ann.get_n_items()}')
        ann = AnnoyIndex(args.dim, 'angular')
        
    # 修改idx2feature，然后重新建库，最后覆盖保存. 后两部解耦出去。
    ### 0. complete embedding if need
    rm_item_idx_list = set(removeOrUpdateItemList.item_idx_list) & set(idx2feature.keys())
    # new_vec_list = removeOrUpdateItemList.new_vec_list
    ### 1. modify idx2feature
    logger.info(f'len(idx2feature): {len(idx2feature)}')
    for rm_idx in rm_item_idx_list:
        del idx2feature[rm_idx]
    logger.info(f'len(idx2feature): {len(idx2feature)}')
    
    change_history.extend([f'rm idx: {x}' for x in rm_item_idx_list])
    msg = f'Will remove these {len(rm_item_idx_list)} items'
    logger.info(msg)
    return {'message': msg}

@app.post('/ann/update_items')
def update_items(removeOrUpdateItemList: data_class.RemoveOrUpdateItemList):
    """Note: please call build_and_save after you finish your several changing calling"""
    global ann, idx2feature, loaded, change_history
    if loaded:
        # first time add
        loaded = False
        logger.info(f'At begin, ann.get_n_items(): {ann.get_n_items()}')
        ann.unload()
        logger.info(f'After unloading, ann.get_n_items(): {ann.get_n_items()}')
        ann = AnnoyIndex(args.dim, 'angular')
        
    # 修改idx2feature，然后重新建库，最后覆盖保存. 后两部解耦出去。
    assert len(item_idx_list) == len(new_vec_list)
    item_idx_list = removeOrUpdateItemList.item_idx_list
    new_vec_list = removeOrUpdateItemList.new_vec_list
    ### 1. modify idx2feature
    tmp = 0
    for idx, new_vec in zip(item_idx_list, new_vec_list):
        if idx not in idx2feature: 
            continue; tmp += 1
        idx2feature[idx]['bge_emb'] = new_vec
        change_history.append(f'update idx: {idx}')
    
    msg = f'Will update these {len(item_idx_list) - tmp} items'
    logger.info(msg)
    return {'message': msg}
    
        

@app.post('/ann/build_and_save')
def build_and_save():
    global change_history
    ### 2. build
    logger.info(f'len(idx2feature): {len(idx2feature)}')
    logger.info(f'ann.get_n_items(): {ann.get_n_items()}')
    for idx, fea in idx2feature.items():
        ann.add_item(idx, fea['bge_emb'])
    ann.build(args.tree_size)
    logger.info(f'After {len(change_history)} changes, ann.get_n_items(): {ann.get_n_items()}')
    ### 3. overwrite
    logger.info(f'Overwrite ann to {ann_binary_file}, {idx_map_file}')
    ann.save(ann_binary_file)
    with open(idx_map_file, 'w') as f:
        for idx, fea in idx2feature.items():
            f.write(json.dumps({'idx': idx, 'fea': fea}) + '\n')
        
    logger.info(f'Done overwriting')
    res = {'message': 'success', 'change_history': change_history.copy()}
    loaded = True
    change_history = []
    return res




if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=args.port)






        