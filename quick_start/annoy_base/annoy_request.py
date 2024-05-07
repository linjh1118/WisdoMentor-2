import fire
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import utils.store.annoy.annoy_client as annoy_client
from utils.base import *
import numpy as np


client = annoy_client.AnnoyClient("http://localhost:8890")
    
# 1. three way to search
print('===== 1 =====')

print('\n## search from text')
query, top_k ="Knowledge Distillation", 3
res = client.search_from_text(query = query, top_k=top_k)
print('indices:', res['indices'])
print('knowledges:', res['knowledges'])

print('\n## search from vec')
res = client.search_from_vec(query_vec = np.arange(1024).tolist(), top_k=top_k)
print('indices:', res['indices'])
print('knowledges:', res['knowledges'])

print('\n## search from vec')
res = client.search_from_item(item_idx = 11, top_k=top_k)
print('indices:', res['indices'])
print('knowledges:', res['knowledges'])

def show_rel(query = "Knowledge Distillation"):
    top_k = 3
    res = client.search_from_text(query = query, top_k=top_k)
    print('indices:', res['indices'])
    print('knowledges:', res['knowledges'])


# 4. add
print('\n## add doc list')
doc_list = get_lines('/home/chy/dream/linjh/rag/wisdomentor/utils/store/annoy/hello_data/add0_to_database')
print('before add')
show_rel(query='BERT')
client.add_docs(doc_list)

# 5. remove
print('\n## remove items list')
print('all_idx: ', client.show_all_idx())
client.remove_items([4, 11])
print('not build')
show_rel()  # [4, 11, 3]  ==> 会成为啥
print('build')
res = client.build_and_save()
show_rel()
print('all_idx: ', client.show_all_idx())

print('test add')
show_rel(query='BERT')

print('change_history: ', '\t'.join(res['change_history']))


