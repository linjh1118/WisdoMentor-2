import fire
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
from utils.embed_model.bge.BgeVocabExpander import BgeVocabExpander

def main(model_path, converted_model_save_path, new_word_file):
    bge_vocab_expander = BgeVocabExpander(model_path)
    new_word_list = open(new_word_file, 'r').read().strip().split('\n')
    bge_vocab_expander.expand(new_word_list, converted_model_save_path)
    
if __name__ == '__main__':
    fire.Fire(main)

# 使用方式：
# pyrag bge_vocab_expand.py --model_path=/home/chy/dream/LLMs/bge-large-zh-v1.5 --converted_model_save_path=/home/chy/dream/LLMs/bge-large-zh-v1.5_vocab --new_word_file=/home/chy/dream/linjh/rag/wisdomentor/quick_start/bge_finetune/new_words_manual