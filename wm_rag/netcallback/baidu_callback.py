
from llama_index.core.tools.tool_spec.base import BaseToolSpec
from llama_index.core import Document
from typing import Dict,Any,Optional,Tuple,List
from base import net_callback,call_info_save_path
import re
import os
import requests
from bs4 import BeautifulSoup
import scrapy
import trafilatura
from trafilatura import bare_extraction
from scrapy.crawler import CrawlerProcess
import json
current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)
question_baidu_path = os.path.join(current_folder,"question_baidu.text")

def extract_url_content(url):
    downloaded = trafilatura.fetch_url(url)
    content =  trafilatura.extract(downloaded)
    
    return {"url":url, "content":content}
# 百度搜索URL
search_url = "http://www.baidu.com/s"

class baidu_callback(net_callback):
    def __init__(self,max_results: Optional[int] = 3) -> None:
        super().__init__()
        self.max_results = max_results
        # 要抓取的页面范围
        self.start_page = 1
        self.end_page = 1
        self.question_path=question_baidu_path
    def baidu_query(self,query:str)->List[str]:
        urls = []
        for page in range(self.start_page, self.end_page + 1):
            # 构建查询参数，包括搜索关键词和页码
            params = {
                'wd': query,
                'rn': str(self.max_results),  # rn是每页显示结果数的参数
                'pn': str((page - 1) * self.max_results)  # pn是页码参数
            }
            
            # 发送HTTP GET请求到百度
            response = requests.get(search_url, params=params)
            
            # 检查请求是否成功
            if response.status_code == 200:
                # 使用BeautifulSoup解析HTML内容
                soup = BeautifulSoup(response.text, 'html.parser')
                
                # 找到所有的搜索结果链接，注意：选择器需要根据百度页面的实际结构进行调整
                results = soup.find_all('h3', class_='t')
                for result in results:
                    link = result.find('a')
                    if link and link['href'].startswith('http'):
                        urls.append(link['href'])
            else:
                print("请求失败，状态码：", response.status_code)
                break  # 如果请求失败，可以选择终止循环
        print(urls)
        return urls
    def recall(self,
               max_result:Optional[int]=3,
               save_pdf_path:Optional[str]=call_info_save_path
               )->None:
        
        query=input("输入关键词:")
        urls=self.baidu_query(query)
        
        upload_files=[]
        txt_names=[]
        id=1
        for url in urls:
            txt_name=query+"_from_Baidu_"+str(id)
            filename=txt_name+".txt"
            result=extract_url_content(url)
            print(result)
            if result['content'] is not None:
                id+=1
                print(os.path.join(call_info_save_path,filename))
                file_path=os.path.join(call_info_save_path,filename)

                with open(file_path, 'w', encoding='utf-8') as file:
                    file.write(result["content"])
                upload_files.append(file_path)
                txt_names.append(txt_name)
                print(f"{result}")

        self.set_upload_files(upload_files)
        self.set_file_names(txt_names)

if __name__=="__main__":
    tool=baidu_callback(max_results=10)
    tool.recall()
    print(tool.llm_ans())