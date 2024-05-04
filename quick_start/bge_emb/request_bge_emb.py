import fire
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
import utils.embed_model.bge.bge_embed_client as bge_client

def main(query):
    client = bge_client.BGEClient("http://localhost:8888/get_embedding/")
    embedding = client.get_embedding(text = query)
    print('query:', query)
    print(f'embedding:\n{embedding}')
    
if __name__ == '__main__':
    fire.Fire(main)