# RAG框架

``` Python
from langchain_core.document import Document
Document(page_content="content", metadata={"key": "value"})
```

## Service

- 插入Text文档到知识库
- 根据Query生成答案

## App Register

## Config

## Eval

### Recall Eval

- HitRate @10(支持配置)

## Modules

### Spliter

- Base方法

  - split_content

- Character Split

  - split_content

- Recursive Split

  - split_content

### Embedding

- Base方法

  - embed_text
  - embed_document
  - embed_documents

- Bert Embedding (只差在P40服务器上部署一个docker服务)

  - embed_text: request.post("url", json={"key": "value"})
  - embed_document: self.embed_text(document.page_content)
  - embed_documents: [self.embed_document(doc) for doc in documents]

- Bge_m3 Embedding (还没有任何实现，需要实现embeding方法和启动bge embedding服务)

  - embed_text: request.post("url", json={"key": "value"})
  - embed_document: self.embed_text(document.page_content)
  - embed_documents: [self.embed_document(doc) for doc in documents]

### WebSearch (已经基本完成，需要读一下代码重构)

- Base 方法
- Arxiv Callback
- Baidu Callback

### Store

- Base 方法

  - insert document
  - search by embed

- Ann Vector Store (只差在P40服务器上部署一个docker服务)

  - insert document: 必须保证含有向量和id的metadata，其它均为可选字段
  - search by embed

### Rerank

- Base 方法

  - rerank_documents(query, documents)

- Bge_rerank_v2_m3 (还没有任何实现，需要实现rerank方法和启动bge rerank服务)

  - rerank_documents(query, documents)

### Prompt Gen

- Base 方法

  - Prompt Gen: List[Docuemnt] + Query + "..."

### Promp Zip

- Base 方法

  - Prompt Zip

- LLMLingua (需要把天润的代码合并到这里)

  - Prompt Gen: Base实现，raise Error
  - Prompt Zip
  - Response Gen Base实现，raise Error

- LLMLingua2 (需要把天润的代码合并到这里)

  - Prompt Gen: Base实现，raise Error
  - Prompt Zip
  - Response Gen Base实现，raise Error

### Response Gen

- Base 方法

  - Response Gen: request.post("url")

- Qwen_72_Gen (还差一个服务部署在16卡Ascend 910b上，需要跟杨振昊对接)

  - Response Gen: request.post("url")
