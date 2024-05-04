
### 1. 如何进行词表拓展？
``` bash
1. 激活对应环境
$ conda activate Embedding

2. cd 工作目录
$ cd ~/dream/linjh/rag/wisdomentor/quick_start/bge_finetune

3. 运行词表拓充脚本
$ sh -x bge_vocab_expand.sh | tee log_expand

+ pyrag=/home/chy/anaconda3/envs/Embedding/bin/python
+ /home/chy/anaconda3/envs/Embedding/bin/python bge_vocab_expand.py --model_path=/home/chy/dream/LLMs/bge-large-zh-v1.5 --converted_model_save_path=/home/chy/dream/LLMs/bge-large-zh-v1.5_vocab --new_word_file=/home/chy/dream/linjh/rag/wisdomentor/quick_start/bge_finetune/new_words_manual
Special tokens have been added in the vocabulary, make sure the associated word embeddings are fine-tuned or trained.

4. 查看词表拓充后的模型和分词器
$ tree /home/chy/dream/LLMs/bge-large-zh-v1.5_vocab
/home/chy/dream/LLMs/bge-large-zh-v1.5_vocab
├── 1_Pooling
│   └── config.json
├── added_tokens.json
├── config.json
├── config_sentence_transformers.json
├── model.safetensors
├── modules.json
├── README.md
├── sentence_bert_config.json
├── special_tokens_map.json
├── tokenizer_config.json
├── tokenizer.json
└── vocab.txt

1 directory, 12 files
```

### 2. 如何微调BGE模型？
``` bash
1. 激活对应环境, cd 工作目录. (同之前)
$ conda activate Embedding
$ cd ~/dream/linjh/rag/wisdomentor/quick_start/bge_finetune

3. 运行微调脚本
$ sh -x bge_finetune.sh | tee log_finetune

```