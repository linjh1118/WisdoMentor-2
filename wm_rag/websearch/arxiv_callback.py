from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core import Document
from typing import Dict,Any,Optional,Tuple,List
from base import NetCallback,call_info_save_path
import asyncio
import arxiv
import re
import os
current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
question_arxiv_path = os.path.join(current_folder,"question_arxiv.text")
def extract_paper_id(url: str) -> str:
    # 使用正则表达式提取论文编号
    match = re.search(r'http://arxiv.org/pdf/([^/]+)', url)
    if match:
        # 返回匹配到的编号部分
        return match.group(1)
    else:
        # 如果没有匹配到编号，返回 None
        return None

class ArxivCallback(NetCallback):
    def __init__(self,max_results: Optional[int] = 3) -> None:
        super().__init__()
        self.max_results = max_results
        self.question_path=question_arxiv_path
        
    spec_functions = ["arxiv_query"]
    def arxiv_query(self,
                    query: str,
                    sort_by: Optional[str] = "relevance"
                    )->Tuple[List[str],List[str],List[Document],arxiv.Search]:
        """
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
        for result in arxiv.Client().results(search):
            titles.append(result.title)
            urls.append(result.pdf_url)
            results.append(
                Document(text=f"{result.pdf_url}: {result.title}\n{result.summary}")
            )
        return urls,titles,results,search
    
    def recall(self,
               query: str, 
               max_result:Optional[int]=3,
               save_pdf_path:Optional[str]=call_info_save_path
               )->None:
        urls,titles,results,search=self.arxiv_query(query)
        papers = arxiv.Client().results(search)
        upload_files=[]
        pdf_names=[]
        for url,title,result,paper in zip(urls,titles,results,papers):
            pdf_name=extract_paper_id(url)
            filename=pdf_name+".pdf"
            paper.download_pdf(filename=filename,dirpath=save_pdf_path)
            upload_files.append(os.path.join(call_info_save_path,filename))
            pdf_names.append(pdf_name)
            print(f"{result}")
        self.set_upload_files(upload_files)
        self.set_file_names(pdf_names)

if __name__=="__main__":
    tool=ArxivCallback()
    tool.recall()
    print(tool.llm_ans())