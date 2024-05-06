from fastapi import FastAPI, Body
from langchain_community.document_loaders import TextLoader

from embedding import BertEmbedding
from store import AnnStore
from prompt import BasePrompt

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from utils.base import *
logger = set_naive_logger(os.path.join(os.path.dirname(__file__), '..', 'log_cache'), log_name='wm_rag.server')

SERVERIP = "82.157.10.130"
app = FastAPI()

@app.post("/chat", summary="获得回答")
def wm_chat(query: str = Body()):
    global logger
    logger.info('##### Welcome to WisdoMentor-2. Please enjoy. #####'); logger.info('')
    logger.info(f'# 1. query: {query}'); logger.info('')
    
    query_embed = BertEmbedding(f"http://{SERVERIP}", "8888").embed_text(query)
    logger.info(f'# 2. query_embedding[:3]: {[float("{:.3f}".format(x)) for x in query_embed[:3]]}'); logger.info('')
    
    recall_documents = AnnStore(f"http://{SERVERIP}", "8890").search_by_embed(query_embed)
    showcase_top3 = [doc.page_content.replace('\n', ' ') for doc in recall_documents[:3]]
    logger.info(f"# 3. top-3 recall: {showcase_top3}"); logger.info('')
    
    # 临时挤在8888端口，长期在8888也行
    response = BasePrompt(f"http://{SERVERIP}", "8888").prompt_gen(query, recall_documents)
    logger.info(f'# 4. response of WisdoMentor-2:\n{response}')
    return response

@app.post("/addDocToRepo", summary="向知识库中添加内容")
def add_doc_to_repo(router_path: str = Body()):
    document = TextLoader(router_path)
    documents = [document]
    documents_embed = BertEmbedding(f"http://{SERVERIP}", "8888").embed_documents(documents)
    AnnStore(f"http://{SERVERIP}", "8890").add_documents(documents, documents_embed)
    return
