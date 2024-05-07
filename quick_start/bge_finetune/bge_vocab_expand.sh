# pyrag=/home/chy/anaconda3/envs/rag/bin/python
pyrag=/home/chy/anaconda3/envs/Embedding/bin/python
$pyrag bge_vocab_expand.py \
  --model_path=/home/chy/dream/LLMs/bge-large-zh-v1.5 \
  --converted_model_save_path=/home/chy/dream/LLMs/bge-large-zh-v1.5_vocab \
  --new_word_file=/home/chy/dream/linjh/rag/wisdomentor/quick_start/bge_finetune/new_words_manual