# WisdoMentor 2.0 Docker Embedding 服务

## 启动

配置无 sudo 的 docker 似乎不太安全，暂时先保留 sudo

``` shell
cd /home/roo/dream/RAG/docker
sudo docker-compose up -d
```

如果需要强制重新构建，则添加 `--build` 参数

``` shell
sudo docker-compose up -d --build
```

如果不希望在后台运行，则取消 `-d` 参数

``` shell
sudo docker-compose up
```

## Embedding

1. bge-m3:

  - port: 10000
  - path: /get_embedding
  - body: 

  ``` json
  {
    "text": [
      "text1",
      "text2",
      ...
    ],
    "type": "bge" | "bert"
  }
  ```

  - return:

  ``` json
  {
    "embedding": [
      [0.98112, 0.456465, ...],
      [...],
      [...],
      ...
    ]
  }
  ```

  - example:

  ``` shell
  curl -X POST -H "Content-Type: application/json" -d '{"text": ["hello", "world"], "model_type": "bge"}' http://localhost:10000/get_embedding
  ```

  ``` shell
  curl -X POST -H "Content-Type: application/json" -d '{"text": ["hello", "world"], "model_type": "bert"}' http://localhost:10000/get_embedding
  ```

## 结构说明

``` shell
.
├── app
│   ├── main.py
│   ├── models
│   │   ├── BertEmbeddingGetter.py    # bert-base-chinese
│   │   ├── BgeEmbeddingGetter.py     # bge-m3
│   │   ├── EmbeddingGetter.py        # 基类
│   │   ├── EmbeddingModelCreator.py  # 根据传入 type 构建对应的 EmbeddingGetter
│   │   └── __init__.py
│   └── requirements.txt
└── Dockerfile
```
