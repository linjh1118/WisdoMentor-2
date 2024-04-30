from os import walk
import pandas as pd
import os
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import faiss
os.environ['KMP_DUPLICATE_LIB_OK']='True'
def delete_vector_by_index(sentences, sentence_embeddings, index):
    # 删除指定索引的句子和向量
    del sentences[index]
    sentence_embeddings = np.delete(sentence_embeddings, index, axis=0)

    # 保存新的句子和向量
    with open('data/sentences.txt', 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write("%s\n" % sentence)
    np.save("data/sentence_embeddings.npy", sentence_embeddings)

    print("删除句子和向量并更新库完毕")

    return sentences, sentence_embeddings
def replace_vector_by_index(sentences, sentence_embeddings, index, new_sentence,model):
    # 计算新句子的向量表示
    new_sentence_embedding = model.encode([new_sentence])[0]

    # 替换指定索引的句子和向量
    sentences[index] = new_sentence
    sentence_embeddings[index] = new_sentence_embedding

    # 保存新的句子和向量
    with open('data/sentences.txt', 'w', encoding='utf-8') as f:
        for sentence in sentences:
            f.write("%s\n" % sentence)
    np.save("data/sentence_embeddings.npy", sentence_embeddings)

    print("替换句子和向量并更新库完毕")

    return sentences, sentence_embeddings
def print_vector_by_index(sentences, sentence_embeddings, index):#示例：print_vector_by_index(sentences, sentence_embeddings, 483)
    print("句子:", sentences[index])
    print("向量:", sentence_embeddings[index])
def load_data():
    # 从'sentences.txt'中加载句子的文本信息
    with open('data/sentences.txt', 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f]
    # 创建一个包含句子的Pandas DataFrame
    df = pd.DataFrame(sentences, columns=['sentence'])
    # 从'sentence_embeddings.npy'中加载句子的向量表示
    sentence_embeddings = np.load("data/sentence_embeddings.npy")

    # 从'index.faiss'中加载向量索引
    index = faiss.read_index("data/index.faiss")

    print(f"Loaded {len(sentences)} sentences, {sentence_embeddings.shape[0]} embeddings, and {index.ntotal} indexed vectors")

    return df,sentences, sentence_embeddings, index
def add_vector(new_sentence, model):
    # 计算新句子的向量表示
    new_sentence_embedding = model.encode([new_sentence])[0]

    # 添加新句子和向量
    sentences.append(new_sentence)
    sentence_embeddings = np.vstack([sentence_embeddings, new_sentence_embedding])

    # 保存新的句子和向量
    with open('data/sentences.txt', 'a', encoding='utf-8') as f:
        f.write("%s\n" % new_sentence)
    np.save("data/sentence_embeddings.npy", sentence_embeddings)

    # 将新的向量添加到索引中
    index.add(np.array([new_sentence_embedding]))
    faiss.write_index(index, 'data/index.faiss')
   
    print("添加句子和向量并更新库完毕")

    return sentences, sentence_embeddings
def search_vector(query,topK,model):
    search = model.encode([query])##搜索字段
    D, I = index.search(search, topK)
    for idx, sentence_idx in enumerate(I[0]):
        print(f"Rank {idx+1}, Index {sentence_idx}, Sentence: {df['sentence'].iloc[sentence_idx]}")
model = SentenceTransformer('shibing624/text2vec-base-chinese')
print("载入模型完毕")
df,sentences, sentence_embeddings, index = load_data()
search_vector("zheshiyigexinjuzi",10,model)
