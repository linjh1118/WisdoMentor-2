from loader import TextLoader
from splitter import CharacterSplitter, RecursiveSplitter
from embedding import BertEmbedding, BgeEmbedding
from store import AnnStore
from websearch import ArxivCallback, BaiduCallback
from rerank import BgeNormalReranker
from prompt_gen import BasePromptGen
from prompt_zip import BasePromptZip, LongLlmLinguaZipper, LlmLingua2Zipper
from response_gen import BaseResponseGen, QwenResponseGen

LOADER = {"text": TextLoader}

SPLITTER = {"character": CharacterSplitter, "recursive": RecursiveSplitter}

EMBEDDING = {"bert": BertEmbedding, "bge": BgeEmbedding}

STORE = {"ann": AnnStore}

WEBSEARCH = {"arxiv": ArxivCallback, "baidu": BaiduCallback}

RERANK = {"bge_normal": BgeNormalReranker}

PROMPTGEN = {"base": BasePromptGen}

PROMPTZIP = {
    "base": BasePromptZip,
    "longllmlingua": LongLlmLinguaZipper,
    "llmlingua2": LlmLingua2Zipper,
}

RESPONSEGEN = {"base": BaseResponseGen, "qwen": QwenResponseGen}
