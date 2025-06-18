# embedding.py
from transformers import AutoTokenizer, AutoModel
import torch 
import numpy as np
import pickle
import os
from .utils import pad_and_truncate

class PhoBertEmbedding:
    """
    PhoBERT embedding extractor for Vietnamese text
    """
    def __init__(self, max_seq_len, pretrained_bert_name='vinai/phobert-base', device='cuda'):
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_name)
        self.model = AutoModel.from_pretrained(pretrained_bert_name)
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model.to(self.device)
        self.model.eval()
        self.max_seq_len = max_seq_len
        self.embedding_dim = self.model.config.hidden_size
    
    def text_to_sequence(self, text, padding='post', truncating='post'):
        # PhoBERT uses BPE tokenizer that requires a specific pre-processing step
        tokens = self.tokenizer.encode(text, add_special_tokens=True)
        return pad_and_truncate(tokens, self.max_seq_len, padding=padding, truncating=truncating)
    
    def get_embedding_matrix(self, word2idx, dat_fname):
        """
        Build embedding matrix using PhoBERT for vocabulary
        """
        if os.path.exists(dat_fname):
            print(f'Loading embedding matrix from: {dat_fname}')
            embedding_matrix = pickle.load(open(dat_fname, 'rb'))
        else:
            print('Creating new embedding matrix using PhoBERT...')
            embedding_matrix = np.zeros((len(word2idx) + 2, self.embedding_dim))
            
            # Extract embeddings for each word in vocabulary
            with torch.no_grad():
                for word, idx in word2idx.items():
                    encoded = self.tokenizer(word, return_tensors='pt')
                    encoded = {k: v.to(self.device) for k, v in encoded.items()}
                    outputs = self.model(**encoded)
                    # Use the [CLS] token embedding or average of all token embeddings
                    word_embedding = outputs.last_hidden_state.mean(dim=1).cpu().numpy()[0]
                    embedding_matrix[idx] = word_embedding
                    
            print(f'Saving embedding matrix to: {dat_fname}')
            pickle.dump(embedding_matrix, open(dat_fname, 'wb'))
            
        return embedding_matrix
    
def initialize_embeddings(tokenizer, embed_dim, embedding_filepath):
    """
    Initialize embeddings using PhoBERT for the given tokenizer
    """
    phobert_embedding = PhoBertEmbedding(max_seq_len=tokenizer.max_seq_len)
    embedding_matrix = phobert_embedding.get_embedding_matrix(tokenizer.word2idx, embedding_filepath)
    return embedding_matrix