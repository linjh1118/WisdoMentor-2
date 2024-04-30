import os
from os import walk
import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# 设置数据目录
dataDir = "./data/output"

# 获取数据目录下的所有文件
allFiles = next(walk(dataDir), (None, None, []))[2]

# 加载原始数据
frames = []
unique_sentences = set()  # Set to store unique sentences
for i in range(len(allFiles)):
    file = allFiles[i]
    print(file)
    df = pd.read_csv("./data/output/"+file, sep="`", header=None, names=["sentence"])
    # Filter out duplicate sentences
    df = df[~df['sentence'].isin(unique_sentences)]
    unique_sentences.update(df['sentence'])
    frames.append(df)

# 将所有数据合并为一个DataFrame
df = pd.concat(frames, axis=0, ignore_index=True)

# 加载模型，将数据进行向量化处理
model = SentenceTransformer('shibing624/text2vec-base-chinese')
sentences = df['sentence'].tolist()
sentence_embeddings = model.encode(sentences)
# Remove duplicate sentence embeddings and corresponding sentences
unique_embeddings, unique_sentences = np.unique(sentence_embeddings, axis=0, return_index=True)
sentence_embeddings = unique_embeddings
sentences = [sentences[i] for i in unique_sentences]
# 将向量处理结果存储
save_file = "data/sentence_embeddings.npy"
np.save(save_file, sentence_embeddings)

# 将句子存储到sentences.txt文件中
with open("data/sentences.txt", "w", encoding="utf-8") as f:
    for sentence in sentences:
        f.write(sentence + "\n")

# 创建faiss索引
index = faiss.IndexFlatL2(sentence_embeddings.shape[1])
index.add(sentence_embeddings)

# 保存faiss索引到文件
index_file = "data/index.faiss"
faiss.write_index(index, index_file)

# 获取文件大小并打印
file_size = os.path.getsize(save_file)
print("%7.3f MB" % (file_size/1024/1024))
