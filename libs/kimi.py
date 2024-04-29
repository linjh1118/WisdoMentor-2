import asyncio
import os
import time
import argparse

from libs.conversation import Conversation
from libs.file import File

question_path=r"C:\Users\cgj\Desktop\Python项目\rag\kimi(1)\kimi\QAs\2207.07051_out.txt"
out_put_path=r"C:\Users\cgj\Desktop\Python项目\rag\kimi(1)\kimi\QA"
async def kimi_pdf_res(upload_file:str,title:str,question_file:str=question_path,output_file:str=out_put_path):
    
    if os.path.isdir(upload_file):
        upload_files = os.listdir(upload_file)
    else:
        upload_files = [upload_file]

    if os.path.isdir(question_file):
        question_files = os.listdir(question_file)
    else:
        question_files = [question_file]

    output_name = output_file.strip().replace("\\", "/").split("/")[-1]

    if not "." in output_name:
        if not os.path.exists(output_file):
            os.makedirs(output_file)

    if os.path.isdir(question_file):
        if len(upload_files) != len(question_files):
            print("文件数量不一致")
            return

    if output_file == "":
        output_file = upload_file

    with open(f"{question_file}", "r", encoding="utf-8") as f:
            questions = f.readlines()
    # print(len(questions),"\n\n\n")
    for i in range(len(upload_files)):
        conv = Conversation()
        print("-" * 20)
        print("创建对话...")
        conv_id = await conv.create_conversation()
        print("成功创建对话" + conv_id)
        print("-" * 20)
        print(f"处理文件{upload_files[i].strip()}")
        file = File(file_path=upload_file)
        try:
            file_id = await file.upload_file()  # 尝试上传文件
            if file_id:
                print("文件上传成功，file_id:", file_id)
            else:
                print("文件上传失败，未获取到有效的file_id。")
        except Exception as e:
            print("文件上传过程中发生错误：", e)
        time.sleep(5)
        for k in range(len(questions)):
            print("-" * 20)
            print(
                f'等待Kimi回复文件"{upload_files[i]}"的问题"{questions[k].strip()}"...'
            )
            attempts = 0
            while attempts < 5:
                ans = await conv.do_conversation(
                    file_id=[file_id],
                    message=[{"role": "user", "content": f"{questions[k].strip()}"}],
                )
                if ans.strip().startswith("您似乎没有提供") or ans.strip().startswith(
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
            with open(f"{output_file}\\{output_file_name}", "a", encoding="utf-8") as f:
                f.write(f"Q:{questions[k].strip()}\nA:{ans.strip()}\n")
        await conv.delete_conversation()
    print(f"请检查路径{output_file.strip()}!")

