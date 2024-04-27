class WMDocSpliter:
    """  """
    def __init__(self, doc):
        self.doc = doc
        self.doc_len = len(doc)
        self.doc_pos = 0
    
    def chunk_spliter(self, chunk_size):
        return NotImplementedError
    
    def sentence_spliter(self):
        return NotImplementedError
    
    def intent_spliter(self):
        return NotImplementedError
    
    
    