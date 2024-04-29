# 导入必要的库
import os
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core import Settings, SimpleDirectoryReader, VectorStoreIndex
from llama_index.llms.ollama import Ollama
from llama_index.core.postprocessor import LongContextReorder
from llama_index.core.response.notebook_utils import display_response

#os.environ["OPENAI_API_KEY"] = "sk-"
# from llama_index.llms.openai import OpenAI

# 设置模型和环境变量
# #Settings.llm = OpenAI(model="gpt-3.5-turbo-instruct", temperature=0.1)
llm = Ollama(model="gemma:2b", temperature=0.1)
Settings.llm = llm
Settings.embed_model = HuggingFaceEmbedding(model_name="D:/llama-index/llama_index/Rerank/bge-base-en-v1.5")

# 从文件中加载文档
documents = SimpleDirectoryReader("./data/paul_graham/").load_data()

# 创建索引
index = VectorStoreIndex.from_documents(documents)

# 设置重排序
reorder = LongContextReorder()
reorder_engine = index.as_query_engine(node_postprocessors=[reorder], similarity_top_k=5)
base_engine = index.as_query_engine(similarity_top_k=5)

# 查询并显示结果
base_response = base_engine.query("Did the author meet Sam Altman?")
display_response(base_response)

reorder_response = reorder_engine.query("Did the author meet Sam Altman?")
display_response(reorder_response)

# 检查阶差
print(base_response.get_formatted_sources())
print(reorder_response.get_formatted_sources())