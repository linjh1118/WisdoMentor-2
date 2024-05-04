import fire
from utils import base
from transformers import AutoModel, AutoTokenizer

def main(query, llm, doc_dir, vector_persist_file=None):
    logger = base.set_naive_logger()
    logger.info(f"query: {query}")
    logger.info(f"llm: {llm}")
    logger.info(f"doc_dir: {doc_dir}")
    logger.info(f"vector_persist_file: {vector_persist_file}")
    # raise NotImplementedError()
    return True


import os
from WMEmbModel import WMEmbModel

if __name__ == "__main__":
    base_dir = os.path.dirname(__file__)
    
    data_dir = os.path.join(base_dir, "data")
    weight_dir = os.path.join(base_dir, "weights")
    wegiht_path = os.path.join(weight_dir, "your_weight")
    
    embModel = WMEmbModel(wegiht_path)
    
    embModel.split(os.path.join(data_dir, "your.txt"), data_dir)
    
    embModel.prepare(os.path.join(data_dir, "your_train_nodes.pkl"), os.path.join(data_dir, "your_val_nodes.pkl"), data_dir)
    
    embModel.finetune(
        os.path.join(weight_dir, "your_model_path"), 
        os.path.join(weight_dir, "your_output_path"), 
        os.path.join(data_dir, "your_train_dataset.json"),
        val_dataset_path=os.path.join(data_dir, "your_val_dataset.json"))

if __name__ == '__main__':
    fire.Fire(main)



""" Quick Start: 
pyfac try_emb.py --llm=qwen:7b --doc_dir micro_bio_dir --query=微生物学是什么意思？
"""

