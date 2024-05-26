import httpx
import json

from .utils import Utils
from .consts import FAKE_HEADERS


class Conversation:
    def __init__(self):
        self.conv_id = ""

    async def create_conversation(
        self, refresh_token: str = None, name: str = "test", force_reget: bool = False
    ) -> str:
        """创建会话

        Args:
            refresh_token (str): refresh_token
            name (str, optional): 会话名称 默认为"test"
            force_reget (bool, optional): 是否强制重新获取access_token 默认为false

        Returns:
            str: refresh_token
        """
        access_token, refresh_token = await Utils().get_access_token(
            refresh_token=refresh_token, force_reget=force_reget
        )
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                "https://kimi.moonshot.cn/api/chat",
                json={
                    "name": name,
                    "is_example": False,
                },
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Referer": "https://kimi.moonshot.cn/",
                    **FAKE_HEADERS,
                },
            )
        try:
            response = Utils().check_response(response)
            res = response.json()
            self.conv_id = res.get("id")
            return self.conv_id
        except:
            if not force_reget:
                print("创建会话失败...尝试重新获取access_token...")
                return await self.create_conversation(refresh_token, name, True)
            else:
                print("创建会话失败...尝试重新获取refresh_token...")
                self.refresh_token = await Utils().get_refresh_token(True)
                return await self.create_conversation(self.refresh_token, name, True)

    async def delete_conversation(
        self, refresh_token: str = None, force_reget: bool = False
    ) -> None:
        """删除会话

        Args:
            refresh_token (str): refresh_token 默认为None,从Utils中获取
            force_reget (bool, optional): 是否强制重新获取token 默认为False

        Returns:
            None
        """
        access_token, refresh_token = await Utils().get_access_token(
            refresh_token=refresh_token, force_reget=force_reget
        )

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.delete(
                f"https://kimi.moonshot.cn/api/chat/{self.conv_id}",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Referer": f"https://kimi.moonshot.cn/chat/{self.conv_id}",
                    **FAKE_HEADERS,
                },
            )
        try:
            response = Utils().check_response(response)
            print("成功删除对话 " + self.conv_id)
        except:
            if not force_reget:
                print("删除会话失败...尝试重新获取access_token")
                await self.delete_conversation(force_reget=True)
            else:
                print("删除会话失败...尝试重新获取refresh_token...")
                self.refresh_token = await Utils().get_refresh_token(True)
                await self.delete_conversation(force_reget=True)

    async def do_conversation(
        self,
        file_id: list[str],
        message: list[str],
        refresh_token: str = None,
        force_reget: bool = False,
    ) -> str:
        """_summary_

        Args:
            file_id (list[str]): 文件列表
            message (list[str]): 消息列表
            refresh_token (str, optional): refresh_token. 默认为None,从Utils中获取
            force_reget (bool, optional): 是否强制重新获取token 默认为False

        Returns:
            str: Kimi的回答
        """
        access_token, refresh_token = await Utils().get_access_token(
            refresh_token=refresh_token, force_reget=force_reget
        )

        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f"https://kimi.moonshot.cn/api/chat/{self.conv_id}/completion/stream",
                json={
                    "messages": message,
                    "refs": file_id,
                    "use_search": False,
                },
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Referer": f"https://kimi.moonshot.cn/chat/{self.conv_id}",
                    **FAKE_HEADERS,
                },
                timeout=120000,
            )

        try:
            response = Utils().check_response(response)
            print("成功完成对话 " + self.conv_id)
            answer = await self.handle_stream(response)
            return answer
        except:
            if not force_reget:
                print("对话失败...尝试重新获取access_token")
                return await self.do_conversation(file_id, message, force_reget=True)
            else:
                print("对话失败...尝试重新获取refresh_token...")
                refresh_token = await Utils().get_refresh_token(True)
                return await self.do_conversation(file_id, message, force_reget=True)

    async def handle_stream(self, response: httpx.Response) -> str:
        """处理回复流

        Args:
            response (httpx.Response): do_conversation中的response

        Returns:
            str: 处理后的消息
        """
        message = ""
        async for line in response.aiter_lines():
            if line.startswith("data:"):
                data = json.loads(line[len("data:") :].strip())
                event = data.get("event")
                text = data.get("text", "")
                if event == "cmpl":
                    message += text
                elif event == "all_done":
                    return message
                elif event == "ping":
                    pass
