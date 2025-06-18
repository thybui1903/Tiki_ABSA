# sentiment_dataset.py
import json
import torch
from torch.utils.data import Dataset
import numpy as np
from utils.tokenizer import tokenize
from utils.vocab import Vocab
from utils.instance import Instance
from builders.dataset_builder import META_DATASET

class TextSegment:
    """Class for representing a text segment with its position, aspect, and sentiment"""
    def __init__(self, start, end, text, aspect, sentiment, confidence=None):
        self.start = start
        self.end = end
        self.text = text
        self.aspect = aspect
        self.sentiment = sentiment
        self.confidence = confidence if confidence is not None else 1.0
        
    def __repr__(self):
        return f"TextSegment(start={self.start}, end={self.end}, aspect={self.aspect}, sentiment={self.sentiment}, text='{self.text}', confidence={self.confidence})"

@META_DATASET.register()
class SentimentDataset(Dataset):
    """Dataset for ABSA (Aspect-Based Sentiment Analysis) from Vietnamese product reviews"""
    
    def __init__(self, data_path, vocab=None, max_seq_length=256, tokenizer=None, task_type="aspect_sentiment"):
        """
        Args:
            data_path: Path to the JSON data file
            vocab: Vocabulary object (if None, will be built from data)
            max_seq_length: Maximum sequence length for padding/truncation
            tokenizer: Tokenization function
            task_type: Type of task - "aspect_sentiment" (joint), "aspect" (aspect only), or "sentiment" (sentiment only)
        """
        self.data_path = data_path
        self.max_seq_length = max_seq_length
        self.tokenizer = tokenizer if tokenizer else tokenize
        self.task_type = task_type
        
        # Load data
        try:
            with open(data_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Data file not found: {data_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format in {data_path}: {e}")
        
        if not self.data:
            raise ValueError(f"No data found in {data_path}")
        
        # Process aspects and sentiments
        self.aspects = set()
        self.sentiments = set()
        self.aspect_sentiment_pairs = set()
        
        self._extract_labels()
        
        # Sort for consistency
        self.aspects = sorted(list(self.aspects))
        self.sentiments = sorted(list(self.sentiments))
        self.aspect_sentiment_pairs = sorted(list(self.aspect_sentiment_pairs))
        
        # Create mappings based on task type
        self._create_label_mappings()
        
        # Initialize or use provided vocabulary
        if vocab is None:
            self.vocab = Vocab()
            self._build_vocab()
        else:
            self.vocab = vocab
            
        # Process data
        self.processed_data = self._process_data()
        
        print(f"Dataset loaded: {len(self.processed_data)} samples, {len(self.labels)} labels")
    
    def _extract_labels(self):
        """Extract all unique aspects, sentiments, and their combinations from data"""
        for item in self.data:
            if "labels" not in item or not item["labels"]:
                continue
                
            for label_info in item["labels"]:
                if len(label_info) < 4:
                    continue
                    
                try:
                    start, end, text, label = label_info[:4]
                    
                    # Parse aspect#sentiment format
                    if '#' in label:
                        aspect, sentiment = label.split('#', 1)
                        self.aspects.add(aspect.strip())
                        self.sentiments.add(sentiment.strip())
                        self.aspect_sentiment_pairs.add(label.strip())
                    else:
                        # If no '#', treat as aspect only
                        self.aspects.add(label.strip())
                        self.aspect_sentiment_pairs.add(label.strip())
                        
                except (ValueError, IndexError) as e:
                    print(f"Warning: Error parsing label {label_info}: {e}")
                    continue
    
    def _create_label_mappings(self):
        """Create label mappings based on task type"""
        if self.task_type == "aspect_sentiment":
            self.labels = self.aspect_sentiment_pairs
        elif self.task_type == "aspect":
            self.labels = self.aspects
        elif self.task_type == "sentiment":
            self.labels = self.sentiments
        else:
            raise ValueError(f"Unknown task_type: {self.task_type}. Must be 'aspect_sentiment', 'aspect', or 'sentiment'")
            
        self.label_to_idx = {label: i for i, label in enumerate(self.labels)}
        self.idx_to_label = {i: label for label, i in self.label_to_idx.items()}
        
        if not self.labels:
            raise ValueError("No labels found in the dataset")
        
        print(f"Found {len(self.labels)} unique labels for task type '{self.task_type}'")
        
    def _build_vocab(self):
        """Build vocabulary from the dataset"""
        print("Building vocabulary...")
        
        # Add words to vocabulary from all text segments
        for item in self.data:
            # Add tokens from full text
            if "text" in item:
                tokens = self.tokenizer(item["text"])
                for token in tokens:
                    self.vocab.add_token(token)
            
            # Add tokens from individual segments
            if "labels" in item:
                for label_info in item["labels"]:
                    if len(label_info) >= 3:
                        segment_text = label_info[2]  # text is at index 2
                        tokens = self.tokenizer(str(segment_text))
                        for token in tokens:
                            self.vocab.add_token(token)
        
        # Build the vocabulary
        self.vocab.build_vocab()
        
        # Add labels to vocabulary
        for label in self.labels:
            self.vocab.add_label(label)
            
        print(f"Vocabulary built: {len(self.vocab)} tokens, {self.vocab.num_labels} labels")
    
    def _process_data(self):
        """Process data into items with multiple labels per text for ABSA"""
        processed_items = []
        
        for item in self.data:
            if "text" not in item:
                continue
                
            full_text = item["text"]
            
            if "labels" not in item or not item["labels"]:
                print(f"Warning: No labels found for text: {full_text[:50]}...")
                continue
            
            # Process each label for this text
            for label_info in item["labels"]:
                try:
                    if len(label_info) < 4:
                        continue
                        
                    # Handle format: [start, end, text, label]
                    start, end, segment_text, full_label = label_info[:4]
                    
                    # Parse aspect and sentiment
                    if '#' in full_label:
                        aspect, sentiment = full_label.split('#', 1)
                        aspect = aspect.strip()
                        sentiment = sentiment.strip()
                    else:
                        aspect = full_label.strip()
                        sentiment = "Bỉnh thường"  # Default sentiment
                        full_label = f"{aspect}#{sentiment}"
                    
                    # Skip if label not in our mapping
                    target_label = full_label if self.task_type == "aspect_sentiment" else aspect if self.task_type == "aspect" else sentiment
                    
                    if target_label not in self.label_to_idx:
                        print(f"Warning: Unknown label '{target_label}' for task type '{self.task_type}'")
                        continue
                    
                    # Tokenize the segment text
                    tokens = self.tokenizer(str(segment_text))
                    token_ids = [self.vocab.token_to_idx(token) for token in tokens]
                    
                    # Truncate or pad
                    original_length = len(token_ids)
                    if len(token_ids) > self.max_seq_length:
                        token_ids = token_ids[:self.max_seq_length]
                    else:
                        token_ids = token_ids + [self.vocab.token_to_idx(self.vocab.pad_token)] * (self.max_seq_length - len(token_ids))
                    
                    # Create label vector (one-hot encoding)
                    label_vector = [0] * len(self.labels)
                    label_idx = self.label_to_idx[target_label]
                    label_vector[label_idx] = 1
                    
                    processed_items.append({
                        "token_ids": token_ids,
                        "length": min(original_length, self.max_seq_length),
                        "labels": label_vector,
                        "original_text": str(segment_text),
                        "full_text": full_text,
                        "spans": [(start, end)],
                        "aspect": aspect,
                        "sentiment": sentiment,
                        "confidence": 1.0,
                        "segment": str(segment_text)
                    })
                    
                except Exception as e:
                    print(f"Warning: Error processing label {label_info}: {e}")
                    continue
        
        if not processed_items:
            raise ValueError("No valid data items were processed")
        
        print(f"Processed {len(processed_items)} segments")
        return processed_items
    
    def get_label_statistics(self):
        """Get statistics about label distribution"""
        label_counts = {}
        aspect_counts = {}
        sentiment_counts = {}
        
        for item in self.processed_data:
            # Count combined labels
            try:
                label_idx = item["labels"].index(1)
                label = self.idx_to_label[label_idx]
                label_counts[label] = label_counts.get(label, 0) + 1
            except ValueError:
                pass  # No label found
            
            # Count aspects and sentiments separately
            aspect = item.get("aspect")
            sentiment = item.get("sentiment")
            
            if aspect:
                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
            if sentiment:
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        return {
            "label_counts": label_counts,
            "aspect_counts": aspect_counts,
            "sentiment_counts": sentiment_counts,
            "total_segments": len(self.processed_data)
        }
    
    def __len__(self):
        return len(self.processed_data)
    
    def __getitem__(self, idx):
        if idx >= len(self.processed_data):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self.processed_data)}")
            
        item = self.processed_data[idx]
        
        # Create instance with additional ABSA-specific information
        instance = Instance(
            input_ids=torch.tensor(item["token_ids"], dtype=torch.long),
            length=item["length"],
            labels=torch.tensor(item["labels"], dtype=torch.float),
            original_text=item["original_text"]
        )
        
        # Add ABSA-specific attributes
        instance.full_text = item["full_text"]
        instance.spans = item["spans"]
        instance.aspect = item["aspect"]
        instance.sentiment = item["sentiment"]
        instance.segment = item["segment"]
        instance.confidence = item["confidence"]
        
        return instance
    
    def get_examples_by_label(self, label, max_examples=5):
        """Get example texts for a specific label"""
        examples = []
        for item in self.processed_data:
            try:
                label_idx = item["labels"].index(1)
                if self.idx_to_label[label_idx] == label:
                    examples.append({
                        "text": item["original_text"],
                        "aspect": item["aspect"],
                        "sentiment": item["sentiment"],
                        "confidence": item["confidence"]
                    })
                    if len(examples) >= max_examples:
                        break
            except ValueError:
                continue  # No label found
        return examples