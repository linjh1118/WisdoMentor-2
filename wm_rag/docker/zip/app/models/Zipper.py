from typing import List
from abc import ABC, abstractmethod
import os
import threading


class Zipper(ABC):

    _instance = None
    _lock = threading.Lock()

    def __new__(cls, *args, **kwargs):
        if not cls._instance:
            with cls._lock:
                if not cls._instance:
                    cls._instance = super(Zipper, cls).__new__(cls)
                    cls._instance.__initialized = False
        return cls._instance

    def __init__(self, weight_path: str = None) -> None:
        if self.__initialized:
            return
        if weight_path is None:
            weight_path = os.path.join(
                os.path.dirname(__file__), "..", "weights", "Llama-3-8b"
            )
        self.weight_path = weight_path
        self.model = None
        self.lock = threading.Lock()
        self.last_access_time = None
        self.timeout = 1800
        self.timer = None
        self.__initialized = True

    @abstractmethod
    def _load_model(self) -> None:
        raise NotImplementedError(
            "'_load_model' method must be overridden in a subclass"
        )

    def _unload_model(self) -> None:
        with self.lock:
            self.model = None

    def _reset_timer(self) -> None:
        if self.timer:
            self.timer.cancel()
        self.timer = threading.Timer(self.timeout, self._unload_model)
        self.timer.start()

    @abstractmethod
    def zip(
        self, docs: List[str], query: str, rate: float = 0.5, instruction: str = ""
    ) -> str:
        raise NotImplementedError("'zip' method shuold be overridden in a subclass")
