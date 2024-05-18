import os
import time
import httpx
import json
from typing import Tuple

from libs.consts import FAKE_HEADERS, ACCESS_TOKEN_EXPIRES


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class Utils:

    def __init__(self):
        self.refresh_token = ""
        self.access_token = ""
        self.access_expires = None

    async def get_refresh_token(self, force_reget: bool = False) -> str:
        """获取refresh_token

        Returns:
            str: refresh_token
        """
        print("-" * 20)
        print("获取refresh_token...")

        if not os.path.exists(f"{os.path.dirname(__file__)}/refresh_token"):
            with open(f"{os.path.dirname(__file__)}/refresh_token", "w") as f:
                f.write("")
        elif not force_reget:
            with open(f"{os.path.dirname(__file__)}/refresh_token", "r") as f:
                self.refresh_token = f.read()
                return self.refresh_token

        # 通过手机号调用发送验证码API
        phone = input("请输入手机号：")

        if len(phone) != 11:
            print("手机号格式不正确")
            return await Utils.get_refresh_token()

        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                "https://kimi.moonshot.cn/api/user/sms/verify-code",
                json={
                    "action": "register",
                    "phone": phone,
                },
            )
        try:
            response = Utils().check_response(response)
        except:
            print("获取验证码失败，请重试")
            return await Utils().get_refresh_token()

        # 通过验证码进行验证登录，获取refresh_token
        # 最多尝试五次

        attempts_time = 0
        while attempts_time < 5:
            verify_code = input("请输入验证码：")
            async with httpx.AsyncClient(timeout=15) as client:
                response = await client.post(
                    "https://kimi.moonshot.cn/api/user/register/trial",
                    json={
                        "phone": phone,
                        "verify_code": verify_code,
                        "wx_user_id": None,
                    },
                )

            try:
                response = Utils().check_response(response)
                data = response.json()
                self.refresh_token = data.get("refresh_token")
                with open(
                    f"{os.path.dirname(__file__)}/refresh_token", "w", encoding="utf-8"
                ) as f:
                    f.write(data.get("refresh_token"))
                return data.get("refresh_token")
            except:
                print("验证失败，请重新输入验证码")
                attempts_time += 1

    async def get_access_token(
        self, refresh_token: str = None, force_reget: bool = False
    ) -> Tuple[str, str]:
        """获取access_token

        Args:
            refresh_token (str, optional): refres_token 默认为None
            force_reget (bool, optional): 是否强制重新获取 默认为False

        Returns:
            Tuple[str, str]: access_token, refresh_token
        """
        if (
            not force_reget
            and self.access_token != ""
            and time.time() < self.access_expires
        ):
            # 不强制重新获取，且之前重新获取过 and not expires，则直接返回
            return refresh_token, self.access_token

        if refresh_token is None:
            refresh_token = self.refresh_token

        if refresh_token == "":
            refresh_token = await self.get_refresh_token()

        async with httpx.AsyncClient(timeout=15) as client:
            response = httpx.get(
                "https://kimi.moonshot.cn/api/auth/token/refresh",
                headers={
                    "Authorization": f"Bearer {refresh_token}",
                    "Referer": "https://kimi.moonshot.cn/",
                    **FAKE_HEADERS,
                },
                timeout=15,
            )

            try:
                response = Utils().check_response(response)
                res = json.loads(response.content.decode("utf-8"))
                access_token = res.get("access_token")
                self.access_token = access_token
                self.access_expires = time.time() + ACCESS_TOKEN_EXPIRES
                return access_token, refresh_token
            except:
                print("获取access_token失败...尝试重新获取refresh_token后重试")
                refresh_token = await Utils.get_refresh_token()
                access_token, refresh_token = await Utils.get_access_token(
                    refresh_token, force_reget
                )
                return access_token, refresh_token

    @staticmethod
    def check_response(response: httpx.Response) -> httpx.Response:
        """检查请求的回复是否正常

        Args:
            response (httpx.Response): 等待被检测的请求

        Raises:
            Exception: 请求status_code不为200抛出

        Returns:
            httpx.Response: 原始请求
        """
        if response.status_code != 200:
            print("请求失败" + response.reason_phrase + " : " + response.text)
            raise Exception("请求失败")
        return response
