from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core import Document
from typing import Dict,Any,Optional
from libs.kimi import kimi_pdf_res
import asyncio
import time
import arxiv
import io
import os
import re

save_path = r"C:\Users\cgj\Desktop\Python项目\rag\kimi(1)\kimi\paper"

class ArxivToolSpec(BaseToolSpec):
    """arXiv tool spec."""

    spec_functions = ["arxiv_query"]

    def __init__(self, max_results: Optional[int] = 3):
        self.max_results = max_results

    def arxiv_query(self, query: str, sort_by: Optional[str] = "relevance"):
        """
        A tool to query arxiv.org
        ArXiv contains a variety of papers that are useful for answering
        mathematic and scientific questions.

        Args:
            query (str): The query to be passed to arXiv.
            sort_by (str): Either 'relevance' (default) or 'recent'

        """
        sort = arxiv.SortCriterion.Relevance
        if sort_by == "recent":
            sort = arxiv.SortCriterion.SubmittedDate
        search = arxiv.Search(query, max_results=self.max_results, sort_by=sort)
        results = []
        titles=[]
        urls=[]
        for result in search.results():
            titles.append(result.title)
            urls.append(result.pdf_url)
            results.append(
                Document(text=f"{result.pdf_url}: {result.title}\n{result.summary}")
            )
        return urls,titles,results,search


def extract_paper_id(url: str) -> str:
    # 使用正则表达式提取论文编号
    match = re.search(r'http://arxiv.org/pdf/([^/]+)', url)
    if match:
        # 返回匹配到的编号部分
        return match.group(1)
    else:
        # 如果没有匹配到编号，返回 None
        return None

class retrieve_from_arxiv_kimi_res():
    def __init__(self) -> None:
        pass
    def load_pdf(self,save_pdf_path:str=save_path):
        query=input("输入关键词:")
        # max_result=input("输入最大论文召回数量(不建议一小时超过50条,传言会被封):")
        Arvix=ArxivToolSpec(3)
        urls,titles,results,search=Arvix.arxiv_query(query)
        papers = arxiv.Client().results(search)
        upload_files=[]
        pdf_names=[]
        for url,title,result,paper in zip(urls,titles,results,papers):
            pdf_name=extract_paper_id(url)
            pdf_url=url
            paper.download_pdf(filename=pdf_name+".pdf",dirpath=save_path)
            upload_files.append(save_path+"\\"+pdf_name+".pdf")
            pdf_names.append(pdf_name)
            print(title)
            print(url)
            print(result)
        self.upload_files=upload_files
        self.pdf_names=pdf_names
    def kimi_res_pdf(self):
        for upload_file,pdf_name in zip(self.upload_files,self.pdf_names):
            asyncio.run(kimi_pdf_res(upload_file=upload_file,title=pdf_name))

if __name__=="__main__":
    pipline=retrieve_from_arxiv_kimi_res()
    pipline.load_pdf()
    pipline.kimi_res_pdf()