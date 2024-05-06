import os
import torch
from llama_index.finetuning import SentenceTransformersFinetuneEngine  # 0.10.32中有，但是0.10.34被舍弃
from llama_index.core.evaluation import EmbeddingQAFinetuneDataset

# 将我的trainer基类搬过来.
# 实现几个基础方法即可: init(模型,数据集,画出曲线,保存)
# 最终目标,让外边的人,可以选择好数据集,然后我这边直接训练(我这里再提供一个数据预处理函数)
class BgeTrainDataPreprocesser:
    """
    Preprocess data for BGE model, which consists of two steps:
        1. split documents into chunks
        2. generate [queries, croups, revelant_docs] from chunks
    """
    
    def __init__(
        self,
    ):
        raise NotImplementedError('This class is not implemented yet')

    def preprocess(self):
        # 1. split documents into chunks
        spliter = BgeDataSpliter(tokenizer_path=self.weight_path, data_file_path=data_file_path)
        spliter.split(output_path, stop_words, max_len, train_size)
        # 2. generate [queries, croups, revelant_docs] from chunks
        preparer = BgeDataPreparer(train_file_path, val_file_path, output_dir)
        preparer.prepare()

class BgeTrainer():
    def __init__(
        self, 
        model_path, save_path,
        train_dataset_file, val_dataset_file,
        batch_size, loss, epochs, 
        evaluation_steps, use_all_docs
    ):
        # check paths
        path_list = [model_path, save_path, train_dataset_file, val_dataset_file]
        assert all([os.path.exists(x) for x in path_list]), 'Some path does not exist'
        self.model_path, self.save_path = model_path, save_path
        # load dataset
        assert all([x.endswith('.json') for x in [train_dataset_file, val_dataset_file]]), 'Dataset should be json file'
        self.train_dataset = EmbeddingQAFinetuneDataset.from_json(train_dataset_file)
        self.val_dataset = EmbeddingQAFinetuneDataset.from_json(val_dataset_file)
        # other config
        self.batch_size, self.loss, self.epochs = batch_size, loss, epochs
        self.evaluation_steps, self.use_all_docs = evaluation_steps, use_all_docs
        
    def finetune(self):
        os.makedirs(self.save_path, exist_ok=False)
        finetune_engine = SentenceTransformersFinetuneEngine(
            dataset=self.train_dataset, val_dataset=self.val_dataset,
            model_path=self.model_path, model_output_path=self.save_path,
            batch_size=self.batch_size, loss=self.loss, epochs=self.epochs,
            evaluation_steps=self.evaluation_steps, use_all_docs=self.use_all_docs
        )
        finetune_engine.finetune()
        
        
        
        
        
        