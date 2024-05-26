# rag框架需要配置成一个常态化的service rag服务，通过http请求获取服务
from fastapi import FastAPI
from entity import InsertDocParams, GetResponseParams
from langchain_core.documents import Document
from app_register import AppRegister

from logger import RagLogger

service = FastAPI()


# 这里还差embedding模型，需要请求embedding服务，再p40上进行常态化启动，需要封装docker
# 这里还差Ann存储服务，需要请求存储服务，在p40上常态化启动，需要封装docker
@service.post("/insertDocToStore", summary="向知识库中插入内容")
async def insert_doc_to_store(insert_doc_params: InsertDocParams) -> None:
    if insert_doc_params.file_type == "stream":
        insert_doc_params.file_content = (
            await insert_doc_params.file_content.read()
        ).decode("utf-8")
    app = AppRegister(insert_doc_params.app_name)
    content = app.load_file(insert_doc_params.file_content)
    contents = app.split_content(content)
    # for content in contents:
    #     print(content.page_content + "\n")
    # print(f"len: {len(contents)}")
    embds = app.get_embeddings(contents)
    for db in app.database:
        db.add_documents(contents, embds)
    return


# 暂时先不接入query扩展，query重写和重排，重排可以考虑bge_reranker_v2_m3，它的效果很好
# 已经接入重排
@service.post("/getResponseFromLLM", summary="根据query获取语言模型回答")
def get_response_from_llm(get_response_params: GetResponseParams) -> str:
    RagLogger().get_logger().info(f"get_response_params: {get_response_params}")
    app = AppRegister(get_response_params.app_name)
    query_embded = app.get_embedding(get_response_params.query_content)
    # web_contents = app.get_websearch_contents(get_response_params.query_content)
    web_contents = []
    logged_web_contents = [
        web_content.page_content
        for web_content in web_contents
        if web_content.page_content
    ]
    RagLogger().get_logger().info(f"web_contents: {logged_web_contents}")
    repo_contents = app.recall([get_response_params.query_content], [query_embded])
    logged_repo_contents = [
        repo_content.page_content
        for repo_content in repo_contents
        if repo_content.page_content
    ]
    logged_repo_contents = "\n\n".join(logged_repo_contents)
    RagLogger().get_logger().info(f"repo_contents: {logged_repo_contents}")
    contents = web_contents + repo_contents
    prompt = app.get_prompt(get_response_params.query_content, contents)
    RagLogger().get_logger().info(f"prompt_original: {prompt}")
    # prompt_zipped = app.zip_prompt(
    #     get_response_params.query_content,
    #     app.split_content(Document(page_content=prompt)),
    # )
    prompt_zipped = prompt
    RagLogger().get_logger().info(f"prompt_zipped: {prompt_zipped}")
    response = app.get_response(prompt_zipped)
    return response


# from langchain_text_splitters import NLTKTextSplitter
