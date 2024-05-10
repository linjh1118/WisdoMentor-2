import requests

class AnnoyClient:
    def __init__(self, url):
        self.url = url

    def search_from_text(self, query: str, top_k: int):
        data = {"query": query, "top_k": top_k}
        postfix = '/ann/search'
        response = requests.post(self.url + postfix, json=data)
        return response.json()

    def search_from_vec(self, query_vec: list[float], top_k: int):
        data = {"query_vec": query_vec, "top_k": top_k}
        postfix = '/ann/search'  # same url as upper
        response = requests.post(self.url + postfix, json=data)
        return response.json()
    
    # 为了方便三路，使用不同的重排，所以这里并不【以一个字典为参数，进而三个函数化为一个】。
    def search_from_item(self, item_idx: int, top_k: int):
        data = {"item_idx": item_idx, "top_k": top_k}
        postfix = '/ann/search'  # same url as upper
        response = requests.post(self.url + postfix, json=data)
        return response.json()
    
    def add_docs(self, doc_list: list[str], doc_emb_list: list[list[float]] = None):
        """try use `my utils.base.get_lines()` to `doc_list` directly"""
        data = {"doc_list": doc_list, "doc_emb_list": doc_emb_list}
        postfix = '/ann/add_docs'
        response = requests.post(self.url + postfix, json=data)
        return response.json()
    
    def remove_items(self, item_idx_list: int):
        data = {"item_idx_list": item_idx_list}
        postfix = '/ann/remove_items'
        response = requests.post(self.url + postfix, json=data)
        return response.json()['message']
    
    def update_items(self, item_idx_list: int, new_vec_list: list[list[float]]):
        data = {"item_idx_list": item_idx_list, 'new_vec_list': new_vec_list}
        postfix = '/ann/update_items'
        response = requests.post(self.url + postfix, json=data)
        return response.json()['message']
    
    def build_and_save(self):
        postfix = '/ann/build_and_save'
        response = requests.post(self.url + postfix)
        return response.json()
    
    def show_all_idx(self):
        postfix = '/ann/show_all_idx'
        response = requests.post(self.url + postfix)
        return response.json()
    
    
