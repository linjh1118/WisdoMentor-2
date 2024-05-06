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
index = faiss.read_index(index_file)
index2chunk = json.load(open(index_file + '2chunk', 'r'))

