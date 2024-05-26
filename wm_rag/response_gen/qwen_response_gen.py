import requests
from .base import ResponseGen


class QwenResponseGen(ResponseGen):
    def __init__(self, router_path: str, port: str) -> None:
        super().__init__()
        self.router_path = router_path
        self.port = port
        return

    def response_gen(self, prompt: str) -> str:
        # res = ""  # TODO: 接入qwen:72b
        # return res
        res = requests.post(
            url=f"{self.router_path}:{self.port}/api/generate",
            json={
                "model": "llama3-8b-chinese",
                "prompt": prompt,
                "stream": False,
            },
        )
        if res.status_code != 200:
            return ""
        res = res.json()["response"]
        return res
