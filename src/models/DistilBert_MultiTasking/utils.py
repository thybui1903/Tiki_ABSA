# utils.py
import json
import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from sklearn.metrics import classification_report, f1_score, accuracy_score
from transformers import AutoTokenizer
import re
from config import ModelConfig
import unicodedata

class TextProcessor:
    """Text preprocessing utilities"""
    
    def __init__(self):
        self.tokenizer = None
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize Vietnamese text"""
        # Remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        # Remove special characters but keep Vietnamese characters
        text = re.sub(r'[^\w\s\u00C0-\u1EF9]', ' ', text)
        return text.strip()
    
    def normalize_label(self, label: str) -> str:
        """Normalize Unicode and replace weird '#' variants"""
        label = unicodedata.normalize("NFC", label)
        label = label.replace('＃', '#')  # Replace full-width hashtag
        return label.strip()
    
    def extract_aspects_from_labels(self, labels: List[List]) -> Dict:
        """Extract aspects and sentiments from label format"""
        aspects = []
        sentiments = []
        spans = []
        
        for label in labels:
            start, end, text, aspect_sentiment = label
            aspect_sentiment = self.normalize_label(aspect_sentiment)

            if '#' in aspect_sentiment:
                parts = aspect_sentiment.split('#')
                if len(parts) == 2:
                    aspect, sentiment = parts
                else:
                    print(f"[Warning] Nhãn lỗi định dạng (quá nhiều '#'): {aspect_sentiment}")
                    continue
            else:
                if aspect_sentiment.lower() == 'khác':
                    aspect = "Khác"
                    sentiment = None
                else:
                    print(f"[Warning] Nhãn không hợp lệ (không có '#'): {aspect_sentiment}")
                    continue

            aspects.append(aspect.strip())
            sentiments.append(sentiment.strip() if sentiment else None)
            spans.append((start, end, text))

            
        return {
            'aspects': aspects,
            'sentiments': sentiments,
            'spans': spans
        }

class DataLoader:
    """Data loading and processing utilities"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.text_processor = TextProcessor()
        
    def load_data(self, file_path: str) -> List[Dict]:
        """Load data from JSON file"""
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    
    def prepare_sample(self, sample: Dict) -> Dict:
        """Prepare a single sample for training"""
        text = self.text_processor.clean_text(sample['text'])
        label_info = self.text_processor.extract_aspects_from_labels(sample['labels'])
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Create multi-hot labels for aspects and sentiments
        aspect_labels = self._create_multi_hot_labels(
            label_info['aspects'], self.config.aspect_labels
        )
        sentiment_labels = self._create_multi_hot_labels(
            label_info['sentiments'], self.config.sentiment_labels
        )
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'aspect_labels': torch.tensor(aspect_labels, dtype=torch.float),
            'sentiment_labels': torch.tensor(sentiment_labels, dtype=torch.float),
            'text': text,
            'original_labels': sample['labels']
        }
    
    def _create_multi_hot_labels(self, labels: List[str], all_labels: List[str]) -> List[int]:
        """Create multi-hot encoding for labels"""
        multi_hot = [0] * len(all_labels)
        for label in labels:
            if label in all_labels:
                multi_hot[all_labels.index(label)] = 1
        return multi_hot

class MetricsCalculator:
    """Calculate various metrics for evaluation"""
    
    @staticmethod
    def calculate_multi_label_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                                    labels: List[str]) -> Dict:
        """Calculate metrics for multi-label classification"""
        metrics = {}
        
        # Overall metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['micro_f1'] = f1_score(y_true, y_pred, average='micro')
        metrics['macro_f1'] = f1_score(y_true, y_pred, average='macro')
        metrics['weighted_f1'] = f1_score(y_true, y_pred, average='weighted')
        
        return metrics
    
    @staticmethod
    def calculate_exact_match_ratio(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate exact match ratio for multi-label classification"""
        exact_matches = np.all(y_true == y_pred, axis=1).sum()
        return exact_matches / len(y_true)
    
import numpy as np

def convert_to_json_serializable(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.float32, np.float64)):
        return float(obj)
    if isinstance(obj, (np.int32, np.int64)):
        return int(obj)
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    if isinstance(obj, dict):
        return {k: convert_to_json_serializable(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_to_json_serializable(i) for i in obj]
    return obj  # fallback
