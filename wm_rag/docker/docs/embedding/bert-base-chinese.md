# BERT-base-Chinese

## 资源使用

使用 3 卡，半小时无人调用自动释放加载好的权重文件，因此长时间无人调用或刚启动服务时，第一次调用速度会比较慢

如果需要更改使用的 GPU ，请修改 `docker-compose.yml` 文件中的 `embedding` 服务，修改其中 `NVIDIA_VISIBLE_DEVICES` 的值

``` dockerfile
services:
  ...
  wisdomentor2-embedding:
    ...
    environment:
      - NVIDIA_VISIBLE_DEVICES=3
    ...
  ...
```

权重文件在构建 image 的时候从 HuggingFace 上下载，无需提前下载后拷贝。下载使用 `https://hf-mirror.com/` 镜像

## 实现细节

直接取 `pooler_output` 作为最终结果。

``` python
encoded_input = self.tokenizer(sentences, return_tensors='pt', padding=True, truncation=True)
with no_grad():
    model_output = self.model(**encoded_input)
res = model_output.pooler_output.tolist()
return res
```
