# BGE微调框架

## 微调过程

1. 采集数据，将其整合为一个txt文件
2. 使用`split`函数将其分割为训练集与数据集
3. 使用`prepare`生成`queries`, `croups`与`revelant_docs`对，即生成微调所需数据
4. 调用`finetune`进行微调

参考代码如下：

```Python
import os
from WMEmbModel import WMEmbModel

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    
    data_dir = os.path.join(base_dir, "data")
    weight_dir = os.path.join(base_dir, "weights")
    wegiht_path = os.path.join(weight_dir, "your_weight")
    
    embModel = WMEmbModel(wegiht_path)
    
    embModel.split(os.path.join(data_dir, "your.txt"), data_dir)
    
    embModel.prepare(os.path.join(data_dir, "your_train_nodes.pkl"), os.path.join(data_dir, "your_val_nodes.pkl"), data_dir)
    
    embModel.finetune(
        os.path.join(weight_dir, "your_model_path"), 
        os.path.join(weight_dir, "your_output_path"), 
        os.path.join(data_dir, "your_train_dataset.json"),
        val_dataset_path=os.path.join(data_dir, "your_val_dataset.json"))
    

```

## 词表扩充

1. 准备需要扩充的词表，以一个`list`的形式传入`expand`，并指定输出路径
2. 同上，生成所需训练数据后调用`finetune`方法进行微调，如果之前生成过可以忽略生成，直接微调

参考代码如下:

``` Python
import os

base_dir = os.path.dirname(__file__)

data_dir = os.path.join(base_dir, "data")
weight_dir = os.path.join(base_dir, "weights")

embModel.expand(["your_word1", "yout_word2"], os.path.join(weight_dir, "your_weight"))
    
embModel.finetune(
    os.path.join(weight_dir, "your_weight"), 
    os.path.join(weight_dir, "your_weight"), 
    os.path.join(data_dir, "your_train_dataset.json"),
    val_dataset_path=os.path.join(data_dir, "your_val_dataset.json"))
```

## 具体实现细节

### 分割

按照停止字符（如逗号、句号等）大致按照句子进行分割，之后尝试将其进行tokenize，如果超出限制则忽略。之后按照默认为8：2的比例划分训练集与数据集，将其存储。

### 数据准备

利用`llama_index`的`generate_qa_embedding_pairs`方法直接生成`queries`、`croups`、`revelant_docs`对，默认使用`qwen:14b`，生成后将其存储

### 微调

利用`sentence_transformers`的`SentenceTransformersFinetuneEngine`与上面生成的数据进行微调

### 词表扩展

可以使用`sentencepiece`调用如`wordpiece`、`bpe`等进行初步处理，之后发现其效果一般，还是推荐人工筛选之后再调用`expand`进行扩展。扩展后需要再次调用`finetune`进行微调。

扩展的过程为：扩展`tokenizer`与`embedding_model`，之后添加池化层，将其转为`sentence_transformer`支持的格式，利用`SentenceTransformersFinetuneEngine`进行微调。

## 测试结果

在小数据集（仅有《微生物学》一本教材）的情况下，先微调两个epochs之后再扩展词表的效果与直接扩展词表进行微调的效果相差不大（只微调两个epochs的原因是再继续微调好像发生了过拟合）
