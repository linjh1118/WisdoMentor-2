# rag框架需要配置成一个常态化的service rag服务，通过http请求获取服务
from fastapi import FastAPI
from entity import InsertDocParams, GetResponseParams
from langchain_core.documents import Document
from app_register import AppRegister

service = FastAPI()

# 这里还差embedding模型，需要请求embedding服务，再p40上进行常态化启动，需要封装docker
# 这里还差Ann存储服务，需要请求存储服务，在p40上常态化启动，需要封装docker
@service.post("/insertDocToStore", summary="向知识库中插入内容")
async def insert_doc_to_store(insert_doc_params: InsertDocParams) -> None:
    if insert_doc_params.file_type == "stream":
        insert_doc_params.file_content = (await insert_doc_params.file_content.read()).decode("utf-8")
    app = AppRegister(insert_doc_params.app_name)
    content = app.load_file(insert_doc_params.file_content)
    contents = app.split_content(content)
    return

# 暂时先不接入query扩展，query重写和重排，重排可以考虑bge_reranker_v2_m3，它的效果很好
@service.post("/getResponseFromLLM", summary="根据query获取语言模型回答")
def get_response_from_llm(get_response_params: GetResponseParams) -> str:
    app = AppRegister(get_response_params.app_name)
    query_embed = [0] * 128 # 这里需要调用embedding模型进行
    web_contents = app.get_websearch_contents(get_response_params.query_content)
    repo_contents = [] # 这里需要调用知识库查询服务进行
    contents = web_contents + repo_contents
    prompt = app.get_prompt(get_response_params.query_content, contents)
    prompt_zipped = app.zip_prompt(prompt) # 这里需要合并一下天润的代码，看看天润的代码是否需要做成服务
    response = app.get_response(prompt_zipped)
    return response

from langchain_text_splitters import NLTKTextSplitter
