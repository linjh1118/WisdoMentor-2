import fire
from utils import base

def main(query, llm, doc_dir, vector_persist_file=None):
    logger = base.set_naive_logger()
    logger.info(f"query: {query}")
    logger.info(f"llm: {llm}")
    logger.info(f"doc_dir: {doc_dir}")
    logger.info(f"vector_persist_file: {vector_persist_file}")
    # raise NotImplementedError()
    return True

if __name__ == '__main__':
    fire.Fire(main)



""" Quick Start: 
pyfac try_qa.py --llm=qwen:7b --doc_dir micro_bio_dir --query=微生物学是什么意思？
"""

