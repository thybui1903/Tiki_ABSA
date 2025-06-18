# tokenizer.py
import os
import pickle
import json
from .utils import pad_and_truncate
from .utils import preprocess_sentence

def tokenize(text):
    """Tokenize text into list of tokens using preprocess_sentence"""
    return preprocess_sentence(text)

def build_tokenizer(data_files, max_seq_len, dat_fname):
    """
    Build a tokenizer from the data files or load from cache
    """
    if os.path.exists(dat_fname):
        print(f'Loading tokenizer from: {dat_fname}')
        tokenizer = pickle.load(open(dat_fname, 'rb'))
    else:
        print(f'Building new tokenizer...')
        text = ''
        for fname in data_files:
            with open(fname, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data:
                    # Add the full text
                    text += item['text'] + ' '
                    # Add each aspect text
                    for label_info in item['labels']:
                        aspect_text = label_info[2]
                        text += aspect_text + ' '
        
        tokenizer = Tokenizer(max_seq_len)
        tokenizer.fit_on_text(text)
        pickle.dump(tokenizer, open(dat_fname, 'wb'))
    return tokenizer



class Tokenizer(object):
    """
    Tokenizer class for Vietnamese text
    """
    def __init__(self, max_seq_len, lower=True):
        self.lower = lower
        self.max_seq_len = max_seq_len
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 1

    def fit_on_text(self, text):
        if self.lower:
            text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    def text_to_sequence(self, text, reverse=False, padding='post', truncating='post'):
        tokens = preprocess_sentence(text)
        unknownidx = len(self.word2idx) + 1
        sequence = [self.word2idx.get(w, unknownidx) for w in tokens]
        if not sequence:
            sequence = [0]
        if reverse:
            sequence = sequence[::-1]
        return pad_and_truncate(sequence, self.max_seq_len, padding=padding, truncating=truncating)

