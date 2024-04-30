import os
from typing import List
import pickle
from llama_index.core.schema import BaseNode
from llama_index.core.llms.utils import LLM
from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SimpleNodeParser
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset
from llama_index.llms.ollama import Ollama
from llama_index.finetuning import generate_qa_embedding_pairs


class BgeDataPreparer:
    def __init__(
        self, train_file_path: str, val_file_path: str, output_dir: str
    ) -> None:
        if not os.path.exists(train_file_path):
            raise FileNotFoundError(
                f"Train file path {train_file_path} does not exist."
            )
        if not os.path.exists(val_file_path):
            raise FileNotFoundError(f"Val file path {val_file_path} does not exist.")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        self.train_file_path = train_file_path
        self.val_file_path = val_file_path
        self.output_dir = output_dir

    def prepare(self, llm: LLM = None) -> None:
        if self.train_file_path.endswith(".pkl"):
            train_nodes = self.load_nodes(self.train_file_path)
        else:
            train_reader = SimpleDirectoryReader(input_files=[self.train_file_path])
            train_parser = SimpleNodeParser.from_defaults(chunk_size=512)
            train_nodes = train_parser.get_nodes_from_documents(train_reader)
            self.save_nodes(train_nodes, os.path.join(self.output_dir, "train.pkl"))
        if self.val_file_path.endswith(".pkl"):
            val_nodes = self.load_nodes(self.val_file_path)
        else:
            val_reader = SimpleDirectoryReader(input_files=[self.val_file_path])
            val_parser = SimpleNodeParser.from_defaults(chunk_size=512)
            val_nodes = val_parser.get_nodes_from_documents(val_reader)
            self.save_nodes(val_nodes, os.path.join(self.output_dir, "val.pkl"))
        if llm is None:
            llm = Ollama(model="qwen:14b")
        train_dataset = self.generate_qa_pairs(llm, train_nodes)
        val_dataset = self.generate_qa_pairs(llm, val_nodes)
        train_dataset.save_json(os.path.join(self.output_dir, "train.json"))
        val_dataset.save_json(os.path.join(self.output_dir, "val.json"))

    def generate_qa_pairs(
        self, llm: LLM, nodes: List[BaseNode]
    ) -> EmbeddingQAFinetuneDataset:
        qa_pairs = generate_qa_embedding_pairs(llm, self.train_file_path)
        return qa_pairs

    def save_nodes(self, nodes: List[BaseNode], output_path: str) -> None:
        with open(output_path, "wb") as f:
            pickle.dump(nodes, f)

    def load_nodes(self, input_path: str) -> List[BaseNode]:
        with open(input_path, "rb") as f:
            return pickle.load(f)
