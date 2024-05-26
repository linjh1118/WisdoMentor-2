import sys

sys.path.append(f"{sys.path[0]}/..")

import os

from langchain_core.documents import Document

from splitter.character_splitter import CharacterSplitter
from splitter.recursive_splitter import RecursiveSplitter

file_path = os.path.join(os.path.dirname(__file__), "splitter_test.txt")

character_splitter = CharacterSplitter(chunk_size=500)
recursive_splitter = RecursiveSplitter(chunk_size=500)

with open(file_path, "r") as f:
    content = f.read()

character_docs = character_splitter.split_content(Document(page_content=content))
recursive_docs = recursive_splitter.split_content(Document(page_content=content))

character_res = [doc.page_content for doc in character_docs]
recursive_res = [doc.page_content for doc in recursive_docs]

with open(
    os.path.join(os.path.dirname(__file__), "splitter_test_character.txt"), "w"
) as f:
    f.write("\n\n".join(character_res))

with open(
    os.path.join(os.path.dirname(__file__), "splitter_test_recursive.txt"), "w"
) as f:
    f.write("\n\n".join(recursive_res))
