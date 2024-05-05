<<<<<<< HEAD
class WMGenerator:
    def __init__(self) -> None:
        pass
    
    def generate(self) -> str:
        pass
    
        
=======
from transformers import AutoTokenizer, AutoModel, AutoModelForCausalLM
import torch
import faiss
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

class WMGenerator:
    def __init__(self, embedder_model_name, generator_model_name, database_texts, dimension=768) -> None:
        self.tokenizer_embedder = AutoTokenizer.from_pretrained(embedder_model_name)
        self.model_embedder = AutoModel.from_pretrained(embedder_model_name)
        
        self.dimension = dimension
        self.index = faiss.IndexFlatL2(self.dimension)
        
        self.tokenizer_generator = AutoTokenizer.from_pretrained(generator_model_name)
        self.model_generator = AutoModelForCausalLM.from_pretrained(generator_model_name)
        if self.tokenizer_generator.pad_token is None:
            self.tokenizer_generator.pad_token = self.tokenizer_generator.eos_token
        
        self.database_texts = database_texts
        self.embeddings = self._embed_texts(database_texts)
        self.index.add(np.array(self.embeddings))

    def _embed_texts(self, texts):
        embeddings = []
        for text in texts:
            inputs = self.tokenizer_embedder(text, return_tensors='pt', truncation=True, max_length=512, padding="max_length", return_attention_mask=True)
            with torch.no_grad():
                outputs = self.model_embedder(**inputs)
            embeddings.append(outputs.last_hidden_state[:, 0, :].squeeze().cpu().numpy())
        return embeddings

    def _retrieve_texts(self, query_embedding, k=5):
        # Retrieve texts based on the query embedding
        distances, indices = self.index.search(np.expand_dims(query_embedding, 0), k)
        return [self.database_texts[i] for i in indices[0]]

    def _generate_text(self, prompt, max_new_tokens=50):
        # Generate text based on the prompt
        inputs = self.tokenizer_generator(prompt, return_tensors='pt', padding=True, truncation=True)
        outputs = self.model_generator.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer_generator.decode(outputs[0], skip_special_tokens=True)

    def generate(self, query, k=5) -> str:
        # Generate a response for a query
        query_embedding = self._embed_texts([query])[0]
        retrieved_texts = self._retrieve_texts(query_embedding, k)
        generated_text = self._generate_text(' '.join(retrieved_texts))
        return generated_text
>>>>>>> 31edf5c (Add files via upload)
