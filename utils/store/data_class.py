from pydantic import BaseModel

# prepare data class
class Vector(BaseModel):
    vector: list[float]
    
class QueryAndTopk(BaseModel):
    query: str
    top_k: int

class QueryVecAndTopk(BaseModel):
    query_vec: list[float]
    top_k: int

class IndexId(BaseModel):
    index_id: int
    
class IndexIdAndNewVec(BaseModel):
    index_id: int
    new_vec: list[float]
    

class ItemIdx(BaseModel):
    item_idx: int
    
class QueryAndTopkANN(BaseModel):
    # 前三个填充一个即可。
    query: str = None
    query_vec: list[float] = None
    item_idx: int = None
    top_k: int = 5
    beam_k: int = 100

class RemoveOrUpdateItemList(BaseModel):    
    item_idx_list: list[int]
    new_vec_list: list[list[float]] = None
    
class DocListAndEmb(BaseModel):
    doc_list: list[str]
    doc_emb_list: list[list[float]] = None

    
    