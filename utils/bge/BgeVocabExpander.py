import os
from transformers import AutoTokenizer, AutoModel
from sentence_transformers import SentenceTransformer, models
import torch


class BgeVocabExpander:
    def __init__(self, weight_path: str, enforce_cpu: bool = False) -> None:
        if not enforce_cpu:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        self.tokenizer = AutoTokenizer(weight_path)
        self.model = AutoModel(weight_path).to(self.device)

    def expand(self, new_words: list[str], output_path: str) -> None:
        new_word_embeddings = []

        for word in new_words:
            subwords = self.tokenizer.tokenize(word)
            subword_indices = self.tokenizer.convert_tokens_to_ids(subwords)
            subword_embeddings = self.model.embeddings.word_embeddings(
                torch.tensor(subword_indices).to(self.device)
            )
            average_embedding = subword_embeddings.mean(dim=0)
            new_word_embeddings.append(average_embedding)

        new_word_embeddings = torch.stack(new_word_embeddings)
        self.tokenizer.add_tokens(new_words)
        self.model.resize_token_embeddings(len(self.tokenizer))
        with torch.no_grad():
            self.model.embeddings.word_embeddings.weight[-len(new_words) :].data.copy_(
                new_word_embeddings
            )

        self.save(output_path)
        self.convert(output_path)

    def save(self, output_path):
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)

    def convert(self, output_path):
        embedding_model = models.Transformer(output_path)
        pooling_model = models.Pooling(
            word_embedding_dimension=768,
            pooling_mode_cls_token=True,
            pooling_mode_mean_tokens=False,
            pooling_mode_max_tokens=False,
            pooling_mode_mean_sqrt_len_tokens=False,
            pooling_mode_weightedmean_tokens=False,
            pooling_mode_lasttoken=False,
            include_prompt=True,
        )
        model = SentenceTransformer(modules=[embedding_model, pooling_model])
        model.save(output_path)
