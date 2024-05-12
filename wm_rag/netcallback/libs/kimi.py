import asyncio
import os
import time
import argparse

from libs.conversation import Conversation
from libs.file import File
from base import question_path,ans_path

async def kimi_file_res(upload_file:str,title:str,question_file:str=question_path,output_file:str=ans_path):
    
    if os.path.isdir(upload_file):
        upload_files = os.listdir(upload_file)
    else:
        upload_files = [upload_file]

    output_name = output_file.strip().replace("\\", "/").split("/")[-1]

    with open(f"{question_file}", "r", encoding="utf-8") as f:
        questions = f.readlines()
        question=""
        for que in questions:
            question+=que+'\n'
    
    for i in range(len(upload_files)):
        conv = Conversation()
        print("-" * 20)
        print("创建对话...")
        conv_id = await conv.create_conversation()
        print("成功创建对话" + conv_id)
        print("-" * 20)
        print(f"处理文件{upload_files[i].strip()}")
        file = File(file_path=upload_file)
        time.sleep(5)
        print("-" * 20)
        print(
            f'等待Kimi回复文件"{upload_files[i]}"的问题...'
        )
        attempts = 0
        while attempts < 5:
            try:
                file_id = await file.upload_file()  # 尝试上传文件
                if file_id:
                    print("文件上传file_id:", file_id)
                else:
                    print("文件上传失败，未获取到有效的file_id。")
            except Exception as e:
                print("文件上传过程中发生错误：", e)
            
            ans = await conv.do_conversation(
                file_id=[file_id],
                message=[{"role": "user", "content": f"{question},如果没有提供需总结的文本或内容，请说“很抱歉您似乎没有提供上传的文件”"}],
            )
            if ans.strip().startswith("很抱歉您似乎没有提供上传的文件") or ans.strip().startswith(
                "您好！看起来您可能忘记提供需要我总结的文本或主题。"
            ):
                print("文件上传未完成,等待5秒...")
                time.sleep(5)
                print("重试...")
                attempts += 1
            else:
                break
        if ans.strip() == "":
            k -= 1
            continue
        print("Answer:", ans.strip())
        output_file_name =  title+ "_out.txt"
        output_file=os.path.join(output_file,output_file_name)
        with open(f"{output_file}", "w+", encoding="utf-8") as f:
            f.write(ans)
        await conv.delete_conversation()
        return ans.strip()
    print(f"请检查路径{output_file.strip()}!")

