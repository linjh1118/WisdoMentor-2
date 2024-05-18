from loader import TextLoader
from splitter import CharacterSplitter
from store import AnnStore
from websearch import ArxivCallback, BaiduCallback
from prompt_gen import BasePromptGen
from prompt_zip import BasePromptZip
from response_gen import BaseResponseGen

LOADER = {
    "text": TextLoader
}

SPLITTER = {
    "character": CharacterSplitter
}

STORE = {
    "ann": AnnStore
}

WEBSEARCH = {
    "arxiv": ArxivCallback,
    "baidu": BaiduCallback
}

PROMPTGEN = {
    "base": BasePromptGen
}

PROMPTZIP = {
    "base": BasePromptZip
}

RESPONSEGEN = {
    "base": BaseResponseGen
}

