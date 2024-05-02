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
res, token_num = zipper.long_llmlingua_zip(model=model_path, 
                 context=text,
                 instruction="",
                 question="介绍叶文洁",
                 rate=0.5,
                 force_tokens=["\n"],
                 )
print(res, token_num)
```

### LLMLingua-2

```python
model_path = "./llmlingua-2-xlm-roberta-large-meetingbank"
res, token_num = zipper.llmlingua2_zip(model=model_path, 
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
res, token_num = zipper.llmlingua2_zip(model=model_path, 
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

按照 [Issue #3 · microsoft/LLMLingua](https://github.com/microsoft/LLMLingua/issues/3) 中的回答：可以忽略此警告，在 LLMLingua 中，逐段处理数据，并压缩超出上下文窗口限制的 KV 缓存。

![](./images/issue.png)



# SftGen/DpoGen使用:

## 参数说明

`data_source` 原数据文件夹，默认为./data，现支持处理单个.txt文件或含有若干.txt文件的文件夹

 `output_file_path` sft数据输出路径，默认为./data/output

`llm_model`使用的大语言模型名称。为"gpt"或"ollama"，默认为"ollama"

## 使用示例

类实例化：

```python
sft_gen = WMSftGen(data_source=data_source, output_file_path=output_file_path, llm_model=llm_model)
dpo_gen = WMDpoGen(data_source=data_source, output_file_path=output_file_path, llm_model=llm_model)
```

方法使用：

```python
# 无返回结果，运行成功后在对应的输出文件夹下输出处理后数据
sft_gen.sft_generator()
dpo_gen.dpo_generator()
```

