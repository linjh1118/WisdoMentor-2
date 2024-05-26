# WisdoMentor 2.0 Docker Rerank 服务

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

## Rerank

1. bge-reranker-v2-m3

  - port: 10002

    path: /bge_normal_rerank

    body:

    ```json
    {
      "query": "什么是微生物？",
      "documents": [
        "微生物是生物体内部和周围环境中的大量微生物。",
        "微生物是构成生物体的细胞，它们可以自我复制和分裂。",
        "微生物可以参与生物体的生长、代谢和免疫功能。",
        ...
      ]
    }
    ```

    会对所有传入的文本进行重排

    return:

    ```json
    "reranked_documents": [
      "文本段1",
      "文本段2",
      ...
    ]
    ```

    返回的是按照分数大小非增序排列的文本段

    example:

    ``` python
    query = "什么是微生物？"

    docs = [
        "计算机是一种用于计算的电子设备。",
        "微生物（microorganism）是生活在环境中的微小生物，它们可以对环境产生影响，并参与生物圈的形成。",
        "微生物是生命体中非常微小的个体，它们可以存在于任何地方，从土壤到空气，从海洋到高山。",
        "微生物可以参与各种生物过程，如呼吸作用、光合作用和厌氧呼吸作用。",
        "微生物可以对环境产生影响，如降解污染物、促进土壤肥力、控制病毒传播等。",
        "微生物是生命体中非常重要的组成部分，它们共同构成了生态系统。",
        "计算机科学（Computer Science）是研究计算机系统结构和行为的学科。",
        "计算机科学是人工智能、网络安全、大数据分析等领域的基石。",
    ]

    res = requests.post(
        url="http://localhost:10002/bge_normal_rerank",
        json={
            "query": query,
            "documents": docs,
        }
    ).json()["reranked_documents"]

    print(res)
    ```
