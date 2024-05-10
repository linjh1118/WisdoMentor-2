"""
(rag) chy@chy-desktop:~/dream/linjh/rag/wisdomentor/utils/store/annoy$ tree hello_data/
hello_data/
└── database
0 directories, 1 file

(rag) chy@chy-desktop:~/dream/linjh/rag/wisdomentor/utils/store/annoy$ pyrag annoy_server.py 
generating embedding: 100%|█████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 30.22it/s]
building ann: 100%|█████████████████████████████████████████████████████████████████████████| 20/20 [00:00<00:00, 9162.87it/s]

(rag) chy@chy-desktop:~/dream/linjh/rag/wisdomentor/utils/store/annoy$ tree hello_data/
hello_data/
├── database
├── database.ann
└── database.idx2fea.jsonl
0 directories, 3 files

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
from collections import defaultdict
from tqdm import tqdm

## === idx: 入库编号 === ##
def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dim", default=1024, type=int, help="知识块嵌入维度")
    parser.add_argument("--input_file", default=os.path.join(os.path.dirname(__file__), "hello_data/database"), type=str, help="知识块特征文件（含嵌入）")
    parser.add_argument("--emb_col_pos", default=-1, type=int, help="嵌入在特征文件的那一列")
    parser.add_argument("--tree_size", default=100, type=int, help="tree_size in ann")
    parser.add_argument("--ann_binary_file", default=os.path.join(os.path.dirname(__file__), "hello_data/database.ann"), type=str, help="ann持久化保存对应的二进制文件")  
    
    # 维护一个map: idx -> {'text': 块文本 等各种原始特征}
    args = parser.parse_args()
    return args

args = get_args()
input_lines = get_lines(args.input_file)
idx2feature = defaultdict(dict)

have_emb = args.emb_col_pos != -1
if have_emb:
    emb_list = [x[args.emb_col_pos] for x in input_lines]
else:
    emb_list = [client.get_embedding(text) for text in tqdm(input_lines, desc=f'generating embedding')]
        

ann = AnnoyIndex(args.dim, 'angular')
for idx, text in tqdm(enumerate(input_lines), total=len(input_lines), desc='building ann'):
    # 记录特征
    idx2feature[idx]['text'] = text
    idx2feature[idx]['bge_emb'] = emb_list[idx]
    # 入库
    ann.add_item(idx, emb_list[idx])
    

# build and save
ann.build(args.tree_size)
# print('a')
ann.save(args.ann_binary_file)
with open(args.ann_binary_file.replace(".ann", ".idx2fea.jsonl"), 'w') as f:
    for idx, fea in idx2feature.items():
        f.write(json.dumps({'idx': idx, 'fea': fea}) + '\n')
# ann.unbuild()
# import numpy as np
# new_vec=np.arange(1024) + np.arange(1024)
# ann.add_item(20, new_vec)