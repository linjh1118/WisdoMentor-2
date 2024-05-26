# WisdoMentor 2.0 Docker Zip 服务

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

## Zip

1. 使用 LongLLMLingua

  - port: 10003

    path: /longllmlingua

    body:
    
    ``` json
    {
      "context": ["context1", "context2", ...],
      "question": "question",
      "rate": 0.5,
      "instruction": ""
    }
    ```

    其中，`rate` 与 `instruction` 可选

    return:

    ```json
    {
      "compressed_prompt": "compressed_prompt"
    }
    ```

2. 使用 LLMLingua2

  - port: 10003

    path: /llmlingua2

    body:
    
    ``` json
    {
      "context": ["context1", "context2", ...],
      "question": "question",
      "rate": 0.5,
      "instruction": ""
    }
    ```

    其中，`rate` 与 `instruction` 可选

    return:

    ```json
    {
      "compressed_prompt": "compressed_prompt"
    }
    ```
