#### 0. 在代码中如何调用`BGE_embedding`服务
``` python
import utils.embed_model.bge.bge_embed_client as bge_client
query = '打算转成嵌入的问题'
client = bge_client.BGEClient("http://localhost:8888/get_embedding/")
embedding = client.get_embedding(text = query)

"""
embedding: 
[0.036426108330488205, 0.049767907708883286, 0.00990158412605524, 0.015758026391267776, 0.03356374427676201, -0.038476720452308655, 0.05231282860040665, -0.002907738322392106, 0.02877327986061573]
"""
```


#### 1. 如果调用`BGE_embedding`服务未成功，可参考如下指令
###### 1.1 检查服务是否已经启动（如果已经启动，则不要重复起服务）
``` bash
chy@chy-desktop:~$ ps -ef | grep $(fuser 8888/tcp)

### a1. 如果服务已然被启动，则输出会如下所示
8888/tcp:           
chy      1146563 1146562  5 16:31 pts/19   00:00:14 /home/chy/anaconda3/envs/rag/bin/python embed_model/bge/bge_embed_server.py

### a2. 如果服务未被启动，则输出会如下所示
Usage: grep [OPTION]... PATTERNS [FILE]...
Try 'grep --help' for more information.
```
###### 1.2 启动服务
``` bash
chy@chy-desktop:~/dream/linjh/rag/wisdomentor/quick_start/bge_emb$ sh -x start_bge_emb_server.sh

正常情况下，该指令运行时，会输出如下内容：
auto select gpu-0, sorted_used: [(0, 303), (1, 303), (2, 303), (3, 303), (4, 303), (5, 303), (6, 303), (7, 303)]
2024-05-03 16:31:30,861 INFO     model_path = /home/chy/dream/LLMs/bge-large-zh-v1.5
2024-05-03 16:31:30,862 INFO     tokenizer_path = /home/chy/dream/LLMs/bge-large-zh-v1.5
2024-05-03 16:31:30,862 INFO     device = cuda:0
2024-05-03 16:31:30,862 INFO     port = 8888
INFO:     Started server process [1146563]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8888 (Press CTRL+C to quit)
```
    

#### 2. 启动后，可使用如下代码 测试`BGE_embedding`服务可用性
``` bash
# Way1. python
pyrag request_bge_emb.py "微生物学主要是讲什么的？" > ../log_cache/log_emb

# Way2. curl
text="微生物学主要是讲什么的？" && curl -X POST -H "Content-Type: application/json" -d "{\"text\": \"${text}\"}" http://localhost:8888/get_embedding/ > ../log_cache/log_emb_way2

"""
# 正常情况下，输出日志文件如下所示:

===log_cache/log_emb===
query: 微生物学主要是讲什么的？
embedding:
[0.036426108330488205, 0.049767907708883286, 0.00990158412605524, 0.015758026391267776, 0.03356374427676201, -0.038476720452308655, 0.05231282860040665, -0.002907738322392106, 0.02877327986061573]

===log_cache/log_emb_way2===
{"embedding":[0.036426108330488205,0.049767907708883286,0.00990158412605524,0.015758026391267776,0.03356374427676201,-0.038476720452308655,0.05231282860040665,-0.002907738322392106,0.02877327986061573]}
"""
```