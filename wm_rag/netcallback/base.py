from typing import List,Optional,Tuple
from abc import ABC, abstractmethod
import asyncio
import os

current_file_path = os.path.abspath(__file__)
current_folder = os.path.dirname(current_file_path)

call_info_save_path = os.path.join(current_folder, "net_callback_content")
question_path = os.path.join(current_folder,"question.text")
question_baidu=os.path.join(current_folder,"question_baidu.text")
ans_path = os.path.join(current_folder,"llm_ans")


class net_callback(ABC):
    def __init__(self) -> None:
        super().__init__()
        if not "." in call_info_save_path:
            if not os.path.exists(call_info_save_path):
                os.makedirs(call_info_save_path)
        if not "." in ans_path:
            if not os.path.exists(ans_path):
                os.makedirs(ans_path)
        self.upload_files = None
        self.file_names = None

        return
    
    def set_upload_files(self, upload_files:List[str])->None:
        self.upload_files = upload_files

    def set_file_names(self, file_names:List[str])->None:
        self.file_names = file_names
        
    @abstractmethod
    def recall(self, max_result:Optional[int]=3)->None:
        """#函数必须最后必须
            使用set_upload_files和set_file_names
        给类变量upload_files，file_names进行赋值"""
        raise NotImplementedError("Subclasses must implement this method.")
    
    def llm_ans(self) -> List[str]:
        
        from libs.kimi import kimi_file_res

        if not hasattr(self, 'upload_files') or not self.upload_files:
            raise ValueError("upload_files 没有被正确赋值或为空")

        if not hasattr(self, 'file_names') or not self.file_names:
            raise ValueError("file_names 没有被正确赋值或为空")
        
        LLM_ans=[]
        for upload_file,file_name in zip(self.upload_files,self.file_names):
            LLM_ans.append(asyncio.run(kimi_file_res(upload_file=upload_file,title=file_name,question_file=self.question_path)))
        return LLM_ans