import asyncio
import os
import time
import argparse

from libs.conversation import Conversation
from libs.file import File


async def main():
    parser = argparse.ArgumentParser(description="Kimi unofficial API")

    parser.add_argument(
        "-u", "--upload-file", type=str, help="希望上传的文件（或路径）"
    )
    parser.add_argument(
        "-q", "--question-file", type=str, help="希望提问的文件（或路径）"
    )
    parser.add_argument(
        "-o", "--output-file", type=str, default="", help="希望保存的文件（或路径）"
    )
    parser.add_argument(
        "-t", "--test", type=bool, default=False, help="使用测试文件进行测试"
    )

    upload_file = parser.parse_args().upload_file
    question_file = parser.parse_args().question_file
    output_file = parser.parse_args().output_file

    # 测试使用
    if parser.parse_args().test:
        upload_file = f"{os.path.dirname(__file__)}/test/uploads"
        question_file = f"{os.path.dirname(__file__)}/test/questions"
        output_file = f"{os.path.dirname(__file__)}/test/outputs"

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
        file = File(file_path=upload_file + "/" + upload_files[i])
        file_id = await file.upload_file()
        print("防止文件上传未完成,等待5秒...")
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
            output_file_name = upload_files[i].split(".")[0] + "_out.txt"
            with open(f"{output_file}/{output_file_name}", "a", encoding="utf-8") as f:
                f.write(f"Q:{questions[k].strip()}\nA:{ans.strip()}\n")
        await conv.delete_conversation()
    print(f"请检查路径{output_file.strip()}!")

if __name__ == "__main__":
    asyncio.run(main())
