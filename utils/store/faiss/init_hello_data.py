import os
import faiss
import numpy as np
import json

with open(os.path.join(os.path.dirname(__file__), 'hello_data', 'database'), 'r') as f:
    lines = f.readlines()
lines = [l.rstrip().strip('"') for l in lines if len(l.strip()) > 0]
with open(os.path.join(os.path.dirname(__file__), 'hello_data', 'database1'), 'w') as f:
    f.write('\n'.join(lines))

# import sys
# sys.path.append('/home/chy/dream/linjh/rag/wisdomentor')
# import utils.embed_model.bge.bge_embed_client as bge_client
# client = bge_client.BGEClient("http://localhost:8888/get_embedding/")
# dim = 1024 # bge-large-zh-v1.5

# index = faiss.IndexFlatL2(dim)   # build the index
# base_vec_list = []
# index_2_textual_chunk = {}
# for idx, l in enumerate(lines):
#     embedding = client.get_embedding(text = l)
#     index_2_textual_chunk[idx] = l
#     base_vec_list.append(embedding)
# base_vecs = np.array(base_vec_list)

# print(index.is_trained)
# index.add(base_vecs)
# print(index.ntotal)

# index_file = os.path.join(os.path.dirname(__file__), 'hello_data', 'database.index')
# faiss.write_index(index, index_file)
# with open(index_file + '2chunk', 'w') as f:
#     json.dump(index_2_textual_chunk, f)

# # pyfac init_hello_data.py