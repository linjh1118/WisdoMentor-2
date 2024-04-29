# PromptZip使用

压缩方法：LLMLingua，LongLLMLingua，LLMLingua-2

## 参数说明

`model` 模型名称或路径；LLMLingua，LongLLMLingua使用huggingface或modelscope中的模型；

LLMLingua-2目前只能使用两种：llmlingua-2-xlm-roberta-large-meetingbank 和 llmlingua-2-bert-base-multilingual-cased-meetingbank（LLMLingua-2的small版本）

`context` prompt中的参考内容部分，可以是字符串或列表

`instruction` prompt中的指令部分

`question` prompt中的问题部分

`rate` 压缩率

`force_tokens` 强制保留的token，列表

返回结果：压缩后的prompt，压缩后的token数量

## 使用示例

导入和数据准备

```python
from WMPromptZip import WMPromptZip

zipper = WMPromptZip()

with open("./santi1.txt", "r", encoding="utf-8") as f:
    text = f.read()
```

### LLMLingua

```python
model_path = "E:\\LLMs\\miniCPM-bf16\\miniCPM-bf16"
res, token_num = zipper.llmlingua_zip(model=model_path, 
                 context=text,
                 instruction="",
                 question="介绍叶文洁",
                 rate=0.5,
                 force_tokens=["\n"],
                 )
print(res, token_num)
```

### LongLLMLingua

```python
model_path = "E:\\LLMs\\miniCPM-bf16\\miniCPM-bf16"
res, token_num = wm.long_llmlingua_zip(model=model_path, 
                 context=text,
                 instruction="",
                 question="介绍叶文洁",
                 rate=0.5,
                 force_tokens=["\n"],
                 )
print(token_num)
```

### LLMLingua-2

```python
model_path = "./llmlingua-2-xlm-roberta-large-meetingbank"
res, token_num = wm.llmlingua2_zip(model=model_path, 
                 context=text,
                 instruction="",
                 question="",
                 rate=0.2,
                 force_tokens=["\n"],
                 )
print(res, token_num)
```

### LLMLingua-2-small

```python
model_path = "./llmlingua-2-bert-base-multilingual-cased-meetingbank"
res, token_num = wm.llmlingua2_zip(model=model_path, 
                 context=text,
                 instruction="",
                 question="",
                 rate=0.2,
                 force_tokens=["\n"],
                 )
print(res, token_num)
```

## Tips

可能出现警告：输入长度过大

```
Token indices sequence length is longer than the specified maximum sequence length for this model (193547 > 512). Running this sequence through the model will result in indexing errorodel will result in indexing errors
```

按照 [Issue #3 · microsoft/LLMLingua (github.com)](https://github.com/microsoft/LLMLingua/issues/3) 中的回答：可以忽略此警告，在 LLMLingua 中，逐段处理数据，并压缩超出上下文窗口限制的 KV 缓存。

![](./images/issue.png)
