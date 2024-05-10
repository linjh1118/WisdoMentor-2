# 由于没有进行调试，所以import部分可能会有缺失，需要进行修改
import numpy as np
import json
from pyserini.search import FaissSearcher, LuceneSearcher
from pyserini.search.faiss import AutoQueryEncoder
from hyde import Promptor, OpenAIGenerator, CohereGenerator, HyDE

class HydeWrapper:
    # 在初始化时，HydeWrapper接收generator、encoder、searcher和promptor作为参数，以便使用这些组件来完成查询和结果处理。
    def __init__(self, generator, encoder, searcher, promptor):
        self.generator = generator
        self.promptor = promptor
        self.encoder = encoder
        self.searcher = searcher

    # 该方法用于生成假设文档，首先根据查询使用promptor.build_prompt方法生成提示，然后通过generator.generate方法获取假设文档。
    def generate_hypotheses(self, query):
        prompt = self.promptor.build_prompt(query)
        hypothesis_documents = self.generator.generate(prompt)
        return hypothesis_documents

    # 该方法用于编码HyDE向量，将查询和生成的假设文档编码为向量，并计算其平均值作为HyDE向量。如果假设文档为空，则返回None。
    def encode_hyde_vector(self, query, hypothesis_documents):
        if not hypothesis_documents:
            return None
        
        all_emb_c = [np.array(self.encoder.encode(text)) for text in [query] + hypothesis_documents]
        avg_emb_c = np.mean(all_emb_c, axis=0).reshape((1, len(all_emb_c[0])))
        return avg_emb_c

    # 该方法用于搜索结果，根据编码的HyDE向量使用searcher.search方法搜索结果，并返回前k个结果。
    def search_results(self, hyde_vector, k=10):
        return self.searcher.search(hyde_vector, k=k)

    # 该方法是端到端的搜索方法，生成假设文档、编码HyDE向量和搜索结果的整体流程。首先生成假设文档，然后编码HyDE向量，最后搜索结果并返回前k个结果。如果假设文档为空，则返回None。
    def search_e2e(self, query, k=10):
        hypothesis_documents = self.generate_hypotheses(query)
        hyde_vector = self.encode_hyde_vector(query, hypothesis_documents)
        if hyde_vector is None:
            return None
        return self.searcher.search(hyde_vector, k=k)