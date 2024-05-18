from loader import TextLoader
from splitter import CharacterSplitter
from store import AnnStore
from websearch import ArxivCallback, BaiduCallback
from prompt import BasePrompt

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
    "base": BasePrompt
}

PROMPTZIP = {
    "base": BasePrompt
}

RESPONSEGEN = {
    "base": BasePrompt
}

