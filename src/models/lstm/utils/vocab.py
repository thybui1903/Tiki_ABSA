

# vocab.py
import torch
import json
from collections import Counter
from builders.vocab_builder import META_VOCAB
from typing import List
from .tokenizer import preprocess_sentence

@META_VOCAB.register()
class Vocab(object):
    def __init__(self, config):
        # Initialize separate class labels for ABSA
        self.class_labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        if config is not None:
            self.initialize_special_tokens(config)
            self.make_vocab(config)
        else:
            # Initialize with default special tokens when no config is provided
            self.pad_token = "<pad>"
            self.bos_token = "<bos>"
            self.eos_token = "<eos>"
            self.unk_token = "<unk>"
            
            self.specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
            
            self.pad_idx = 0
            self.bos_idx = 1
            self.eos_idx = 2
            self.unk_idx = 3
            
            # Initialize empty vocabulary
            self.itos = {i: tok for i, tok in enumerate(self.specials)}
            self.stoi = {tok: i for i, tok in enumerate(self.specials)}
            self.max_sentence_length = 0
            
            

    def initialize_special_tokens(self, config) -> None:
        self.pad_token = config.pad_token
        self.bos_token = config.bos_token
        self.eos_token = config.eos_token
        self.unk_token = config.unk_token

        self.specials = [self.pad_token, self.bos_token, self.eos_token, self.unk_token]

        self.pad_idx = 0
        self.bos_idx = 1
        self.eos_idx = 2
        self.unk_idx = 3

    def make_vocab(self, config):
        json_dirs = [config.path.train, config.path.dev, config.path.test]
        counter = Counter()
        self.max_sentence_length = 0
        
        # Separate collections for vocabulary and class labels
        class_labels_set = set()

        for json_dir in json_dirs:
            with open(json_dir, encoding='utf-8') as f:
                data = json.load(f)

            for item in data:
                # Tokenize text chính
                raw_text = item["text"]
                tokenized_text = preprocess_sentence(raw_text)
                counter.update(tokenized_text)

                if self.max_sentence_length < len(tokenized_text):
                    self.max_sentence_length = len(tokenized_text)

                # Process ABSA labels (separate from vocabulary)
                if "labels" in item:
                    for label_info in item["labels"]:
                        # Handle new format: [start, end, text, label]
                        if len(label_info) >= 4:
                            start, end, aspect_text, full_label = label_info[:4]
                            
                            # Add the ABSA label to class labels
                            class_labels_set.add(full_label)
                            
                            # Tokenize aspect text for vocabulary (not labels)
                            aspect_text = str(aspect_text)
                            tokenized_aspect = preprocess_sentence(aspect_text)
                            counter.update(tokenized_aspect)

        # Build vocabulary (for text tokens only)
        min_freq = max(config.min_freq, 1)
        words_and_frequencies = sorted(counter.items(), key=lambda tup: tup[0])
        words_and_frequencies.sort(key=lambda tup: tup[1], reverse=True)

        itos = []
        for word, freq in words_and_frequencies:
            if freq < min_freq:
                break
            itos.append(word)

        itos = self.specials + itos
        self.itos = {i: tok for i, tok in enumerate(itos)}
        self.stoi = {tok: i for i, tok in enumerate(itos)}
        
        # Build class label mappings (separate from vocabulary)
        self.class_labels = sorted(list(class_labels_set))
        self.class_to_idx = {label: i for i, label in enumerate(self.class_labels)}
        self.idx_to_class = {i: label for i, label in enumerate(self.class_labels)}
        
        print(f"Vocabulary size: {len(self.stoi)}")
        print(f"Number of ABSA classes: {len(self.class_labels)}")
        print(f"ABSA classes: {self.class_labels}")

    # Methods for vocabulary tokens
    def token_to_idx(self, token):
        """Convert token to index, return unk_idx if not found"""
        return self.stoi.get(token, self.unk_idx)
    
    def add_token(self, token):
        """Add a token to vocabulary if it doesn't exist"""
        if token not in self.stoi:
            idx = len(self.stoi)
            self.stoi[token] = idx
            self.itos[idx] = token
    
    def add_label(self, label):
        """Add a label to vocabulary (same as add_token for this implementation)"""
        # For backward compatibility, but should use class label methods instead
        self.add_token(label)

    # Methods for ABSA class labels (separate from vocabulary)
    def add_class_label(self, label):
        """Add a class label for ABSA"""
        if label not in self.class_to_idx:
            idx = len(self.class_labels)
            self.class_labels.append(label)
            self.class_to_idx[label] = idx
            self.idx_to_class[idx] = label
    
    def class_label_to_idx(self, label):
        """Convert class label to index"""
        return self.class_to_idx.get(label, -1)  # Return -1 for unknown class
    
    def idx_to_class_label(self, idx):
        """Convert index to class label"""
        return self.idx_to_class.get(idx, "Unknown")
    
    def get_num_classes(self):
        """Get number of ABSA classes"""
        return len(self.class_labels)

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode_sentence(self, sentence: str) -> torch.Tensor:
        """ Turn a sentence into a vector of indices and a sentence length """
        sentence = preprocess_sentence(sentence)
        vec = [self.bos_idx] + [self.stoi[token] if token in self.stoi else self.unk_idx for token in sentence] + [self.eos_idx]
        vec = torch.Tensor(vec).long()
        return vec

    def decode_sentence(self, sentence_vecs: torch.Tensor, join_words=True) -> List[str]:
        '''
            sentence_vecs: (bs, max_length)
        '''
        sentences = []
        for vec in sentence_vecs:
            question = " ".join([self.itos[idx] for idx in vec.tolist() if self.itos[idx] not in self.specials])
            if join_words:
                sentences.append(question)
            else:
                sentences.append(question.strip().split())
        return sentences

    # Backward compatibility methods (now properly separated)
    def idx2label(self, idx):
        """
        For backward compatibility - returns vocabulary token
        Use idx_to_class_label() for ABSA class labels
        """
        return self.itos.get(idx, self.unk_token)
    
    def label2idx(self, label):
        """
        For backward compatibility - returns vocabulary index
        Use class_label_to_idx() for ABSA class labels
        """
        return self.stoi.get(label, self.unk_idx)

    def __eq__(self, other: "Vocab"):
        if self.stoi != other.stoi:
            return False
        if self.itos != other.itos:
            return False
        if hasattr(self, 'class_to_idx') and hasattr(other, 'class_to_idx'):
            if self.class_to_idx != other.class_to_idx:
                return False
        return True

    def __len__(self):
        return len(self.itos)