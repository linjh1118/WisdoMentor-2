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

## Store

1. Annoy

  数据挂载路径： `/home/roo/dream/RAG/data/AnnStore`

  - port: 10001
  
    path: /ann/add_docs
    
    body:

    ```json
    {
      "doc_list": ["doc1", "doc2",...],
      "doc_emb_list": [
        [0.98112, 0.456465, ...],
        [...],
        [...],
        ...
      ]
    }
    ```

      - doc_list： 文档内容
      - doc_emb_list： 文档对应的向量

    return: None

    example:

    ``` python
    sentences = [
      "apple",
      "banana"
    ]
    embs = requests.post(
              url="http://localhost:10000/get_embedding/",
              json={
                  "text": sentences,
                  "model_type": "bert"
              }
    ).json()["embedding"]
    requests.post(
      url="http://localhost:10001/ann/add_docs",
      json={
          "doc_list": sentences,
          "doc_emb_list": embs
      }
    )
    ```

  - port: 10001

    path: /ann/search

    body:

    ```json
    {
      "query_vec": [
        [0.98112, 0.456465, ...],
        [...],
        [...],
        ...
      ],
      "num": 10
    }
    ```

      - query_vec: 查询向量
      - num: 返回结果数量

    return:

    ```json
    {
      "knowledges": [
        ["doc1", "doc2",...],
        [...],
        [...],
        ...
      ]
    }
    ```

    每个元素代表一个 vec 召回的文档

    example:

    ``` python
    search_sentences = [
    "app"
    ]
    search_embs = requests.post(
            url="http://localhost:10000/get_embedding/",
            json={
                "text": search_sentences,
                "model_type": "bert"
            }
    ).json()["embedding"]
    search_res = requests.post(
    url="http://localhost:10001/ann/search",
    json={
        "query_vec": search_embs,
        "num": 2
    }
    ).json()["knowledges"]
    print("search res", search_res)
    ```

  - port: 10001

    path: /ann/get_ids

    body:

    ``` json
    {
      "docs": ["doc1", "doc2", ...]
    }
    ```

    return:

    ``` json
    {
      "ids": [1, -1, ...]
    }
    ```

    如果找到则为对应的 id ，否则为 -1

    example:

    ``` python
    ids = requests.post(
        url="http://localhost:10001/ann/get_ids",
        json={
            "docs": ["apple", "banana"]
        }
    ).json()["ids"]
    ```
  
  - port: 10001

    path: /ann/remove_items

    body:

    ``` json
    {
      "ids": [1, -1, ...]
    }
    ```

    return: None

    example:

    ``` python
    requests.delete(
      url="http://localhost:10001/ann/remove_items",
      json={
          "ids": [-1]
        }
    )
    ```

    可以正确处理 -1 ，查询后不需要筛选

## 结构说明

``` shell
.
├── app
│   ├── AnnStore.py       # Annoy 实现
│   ├── data
│   │   ├── store.ann     # Annoy 索引
│   │   └── store.db      # sqlite 数据
│   ├── main.py
│   └── requirements.txt
├── Dockerfile
└── tests
    └── test.py           # 测试文件
```
