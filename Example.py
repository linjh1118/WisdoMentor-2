import asyncio  # 用于异步操作

# Kimi的文件没有和对话绑定，所以可以先上传文件
from libs.file import File
import time

file_path = r"C:\Users\cgj\Desktop\Python项目\rag\kimi(1)\kimi\QAs\2207.07051_out.txt"
# file_path必传，file_name可选，不传入file_name会自动获取为文件名，如file1.pdf
# file_name需要包含扩展名，后续需要根据file_name推测文件类型
# 一般默认不传入即可，名字没有什么影响
file = File(file_path=file_path, file_name=None)
# 之后上传文件
file_id = asyncio.run(file.upload_file())
time.sleep(5)  # 防止文件上传未完成

# 之后是创建一个对话
from libs.conversation import Conversation

# 创建对话，并获取id
# refresh_token一般不需要手动传入
# name表示对话名字
# force_reget表示是否强制重新获取access_token
# 一般保持默认即可
conv = Conversation(refresh_token=None, name="test", force_reget=False)
# 这个conv_id可以不取
conv_id = asyncio.run(conv.create_conversation())
# 可以调用create_conversation之后这么取出来
conv_id = conv.conv_id

# 之后就可以进行对话了
question = "your question"  # 问题
# 这里需要传入file_id与message
# file_id可以传入多个文件（理论上可以）
# message需要指定role与content，一般想要提问的话role为user,content为问题
ans = asyncio.run(
    conv.do_conversation(
        file_id=[file_id], message=[{"role": "user", "content": question}]
    )
)
if ans.strip().startswith("您似乎没有提供") or ans.strip().startswith(
    "您好！看起来您可能忘记提供需要我总结的文本或主题。"
):
    # 这部分是测试出来的文件未上传完成的两个可能回答，如果发现有新的可以也加到这里
    # 这个时候可以等待后直接重新调用conv.do_conversation来回答同样的问题
    print("文件未上传完成")
