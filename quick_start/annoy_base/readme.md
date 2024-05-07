# ANN向量库服务调用指南

#### 优点：本rag框架封装后的ANN服务已经 支持CURD，支持持久化文件的动态更新，支持更新历史的自动记录
#### 解释：faiss工程改造繁琐，而ANN便于封装改造。所以从之前封装FAISS转变为封装ANN


### 1. 建立初始ANN库
``` bash
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
```


### 2. 启动服务
``` bash
(rag) chy@chy-desktop:~/dream/linjh/rag/wisdomentor/utils/store/annoy$ pyrag annoy_server.py 
2024-05-05 01:56:39,587 INFO     database_dir = hello_data
2024-05-05 01:56:39,587 INFO     database_id = database
2024-05-05 01:56:39,587 INFO     dim = 1024
2024-05-05 01:56:39,587 INFO     tree_size = 100
2024-05-05 01:56:39,587 INFO     port = 8890
INFO:     Started server process [1472559]
INFO:     Waiting for application startup.
2024-05-05 01:56:39,625 INFO     Load Ann from hello_data/database.ann, hello_data/database.idx2fea.jsonl
2024-05-05 01:56:39,634 INFO     Done loading
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8890 (Press CTRL+C to quit)
```

### 3. 调用服务
``` python
import utils.store.annoy.annoy_client as annoy_client
client = annoy_client.AnnoyClient("http://localhost:8890")

==检索==
print('\n## search from text')
query, top_k ="Knowledge Distillation", 3
res = client.search_from_text(query = query, top_k=top_k)
print('indices:', res['indices'])
print('knowledges:', res['knowledges'])
# 支持向量检索，支持item入库号检索
# res = client.search_from_vec(query_vec = np.arange(1024).tolist(), top_k=top_k)
# res = client.search_from_item(item_idx = 11, top_k=top_k)

==增加==
print('\n## add doc list')
doc_list = get_lines('/home/chy/dream/linjh/rag/wisdomentor/utils/store/annoy/hello_data/add0_to_database')
print('before add')
show_rel(query='BERT')
client.add_docs(doc_list)
client.build_and_save()
show_rel(query='BERT')

==删除==
print('\n## remove items list')
print('all_idx: ', client.show_all_idx())
show_rel()
client.remove_items([4, 11])
res = client.build_and_save()
print('change_history: ', '\t'.join(res['change_history']))
show_rel()
```


