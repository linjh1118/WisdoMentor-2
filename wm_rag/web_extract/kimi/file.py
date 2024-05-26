from typing import Tuple
import httpx
from mimetypes import guess_type
import aiofiles


from .consts import FAKE_HEADERS
from .utils import Utils


class File:
    def __init__(self, file_path: str, file_name: str = None):
        self.file_path = file_path
        if file_name is None:
            self.file_name = file_path.replace("\\", "/").strip().split("/")[-1]
        else:
            self.file_name = file_name
        self.url = ""
        self.object_name = ""
        self.file_id = ""

    async def upload_file(self, force_reget: bool = False) -> str:
        """上传文件

        Args:
            force_reget (bool, optional): 是否强制获取token 默认为False

        Returns:
            str: 文件id
        """
        if self.url == "" or self.object_name == "":
            await self.get_pre_sign_url()
        mime_type = guess_type(self.file_name)[0] or "application/octet-stream"
        async with aiofiles.open(self.file_path, "rb") as file:
            file_data = await file.read()
        access_token, refresh_token = await Utils().get_access_token()
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.put(
                self.url,
                content=file_data,
                headers={
                    "Content-Type": mime_type,
                    "Authorization": f"Bearer {access_token}",
                    "Referer": "https://kimi.moonshot.cn/",
                    **FAKE_HEADERS,
                },
            )
        try:
            response = Utils().check_response(response)
        except:
            if not force_reget:
                print("上传文件失败... 尝试重新获取access_token")
                return await self.upload_file(force_reget=True)
            else:
                print("上传文件失败... 尝试重新获取refresh_token")
                await Utils().get_refresh_token(True)
                return await self.upload_file(force_reget=True)

        access_token, refresh_token = await Utils().get_access_token()
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                "https://kimi.moonshot.cn/api/file",
                json={
                    "type": "file",
                    "name": self.file_name,
                    "object_name": self.object_name,
                },
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "Referer": "https://kimi.moonshot.cn/",
                    **FAKE_HEADERS,
                },
            )
        try:
            response = Utils().check_response(response)
            self.file_id = response.json().get("id")
            return self.file_id
        except:
            if not force_reget:
                print("上传文件失败 尝试重新获取access_token")
                return await self.upload_file(force_reget=True)
            else:
                print("上传文件失败 尝试重新获取refresh_token")
                await Utils().get_refresh_token(True)
                return await self.upload_file(force_reget=True)

    async def get_pre_sign_url(
        self, refresh_token: str = None, force_reget: bool = False
    ) -> Tuple[str, str]:
        """获取预签名url与object_name

        Args:
            refresh_token (str, optional): refresh_token 默认为None,从Utils中获取
            force_reget (bool, optional): 是否强制重新获取token 默认为False

        Returns:
            Tuple[str, str]: url, object_name
        """
        access_token, refresh_token = await Utils().get_access_token(refresh_token)
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                "https://kimi.moonshot.cn/api/pre-sign-url",
                json={
                    "action": "file",
                    "name": self.file_name,
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
            self.url = res.get("url")
            self.object_name = res.get("object_name")
            return self.url, self.object_name
        except Exception as e:
            if not force_reget:
                print("获取预签名URL失败... 尝试重新获取access_token")
                return await self.get_pre_sign_url(
                    refresh_token=refresh_token, force_reget=True
                )
            else:
                print("获取预签名URL失败... 尝试重新获取refresh_token")
                refresh_token = await Utils().get_refresh_token(True)
                return await self.get_pre_sign_url(
                    refresh_token=refresh_token, force_reget=True
                )

    async def get_file_id(self):
        return self.file_id
