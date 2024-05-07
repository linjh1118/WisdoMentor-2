import requests

class BGEClient:
    def __init__(self, url):
        self.url = url

    def get_embedding(self, text):
        data = {"text": text}
        response = requests.post(self.url, json=data)
        embedding = response.json()["embedding"]
        return embedding

    

"""
``` python

import utils.embed_model.bge.bge_embed_client as bge_client

client = bge_client.BGEClient("http://localhost:8888/get_embedding/")
embedding = client.get_embedding(text = '微生物学主要是讲什么的？')

print('query:', text)
print(f'embedding:\n{embedding}')
```


``` bash
text="微生物学主要是讲什么的？" && curl -X POST -H "Content-Type: application/json" -d "{\"text\": \"${text}\"}" http://localhost:8888/get_embedding/ > log_emb
```
"""