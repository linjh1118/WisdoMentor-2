import os
from typing import Tuple
import random
import re
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split


class BgeDataSpliter:
    def __init__(self, tokenizer_path: str, data_file_path: str) -> None:
        """初始化BgeDataSpliter

        Args:
            tokenizer_path (str): tokenizer位置
            data_file_path (str): 数据文件位置

        Raises:
            FileNotFoundError: 找不到文件
            e: 未知错误
        """
        if not os.path.exists(tokenizer_path):
            raise FileNotFoundError(f"Tokenizer path {tokenizer_path} does not exist.")
        if not os.path.exists(data_file_path):
            raise FileNotFoundError(f"Data file path {data_file_path} does not exist.")
        if not os.path.isfile(data_file_path):
            raise FileNotFoundError(f"Data file path {data_file_path} is not a file.")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        except Exception as e:
            raise e
        self.data_file_path = data_file_path

    def split(
        self,
        output_path: str,
        stop_words: list[str] = None,
        max_len: int = 512,
        train_size: float = 0.8,
    ) -> Tuple[list[str], list[str]]:
        """将数据文件分割为训练集和验证集，并尝试写入文件

        Args:
            output_path (str): 输出路径
            stop_words (list[str], optional): 停止词 分句使用 默认为None
            max_len (int, optional): 最大长度限制 默认为512.
            train_size (float, optional): 训练集比重 默认为 0.8.

        Returns:
            Tuple[list[str], list[str]]: 训练集数据与验证集数据
        """
        # 读取文件内容
        with open(self.data_file_path, "r") as f:
            data = f.readlines()
        # 过滤空行
        data = [d.strip() for d in data if d.strip()]
        # 定义停止词
        if stop_words is None:
            stop_words = ["。", "！", "？", "；", ".", "?", ";", ":", "\n"]
        # 构建正则表达式
        pattern = "|".join(map(re.escape, stop_words))
        # 按照正则表达式分割文本
        sentence = "".join(data)
        sentences = re.split(pattern, sentence)
        # 去除空行
        sentences = [s.strip() for s in sentences if s.strip()]

        # 按块分割文本，争取每块长度不超过max_len
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            try:
                tokens = self.tokenizer.tokenize(sentence)
            except:
                # 这里似乎tokenize会对超出限制的文本有一个处理，但还是加上这个了
                print(f"{sentence} is too long, ignore it.")
            current_length += len(tokens)
            if current_length > max_len:
                # 当前块已满，将其加入chunks列表
                chunks.append(" ".join(current_chunk))
                current_chunk = []  # 开始新块
                current_length = len(tokens)  # 重置长度
            current_chunk.append(sentence)

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        random.shuffle(chunks)  # 打乱顺序
        train, val = train_test_split(chunks, train_size=train_size)

        try:
            with open(os.path.join(output_path, "train.txt"), "w") as f:
                for t in train:
                    f.write(t + "\n\n")

            with open(os.path.join(output_path, "val.txt"), "w") as f:
                for v in val:
                    f.write(v + "\n\n")
        except Exception as e:
            print(f"Error occurred when writing data to file: {e}, data may be lost.")

        return (train, val)
