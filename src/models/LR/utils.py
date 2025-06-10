import json
import re
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Any
import pickle
import os

class DataLoader:
    def __init__(self):
        self.aspect_encoder = LabelEncoder()
        self.sentiment_encoder = LabelEncoder()
        
    def load_json_data(self, file_path: str) -> List[Dict]:
        """Load dữ liệu từ file JSON"""
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def preprocess_text(self, text: str) -> str:
        """Tiền xử lý văn bản tiếng Việt"""
        # Loại bỏ ký tự đặc biệt, giữ lại chữ cái, số và khoảng trắng
        text = re.sub(r'[^\w\s]', ' ', text)
        # Loại bỏ số
        text = re.sub(r'\d+', '', text)
        # Chuẩn hóa khoảng trắng
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()
    
    def extract_aspects_sentiments(self, data: List[Dict]) -> Tuple[List[str], List[str], List[str]]: 
        """Trích xuất aspect và sentiment từ dữ liệu"""
        texts = []
        aspects = []
        sentiments = []
        
        for item in data:
            text = item['text']
            for label in item['labels']:
                start, end, aspect_text, aspect_sentiment = label
                if '#' in aspect_sentiment:
                    aspect_category, sentiment_label = aspect_sentiment.split('#', 1)
                else:
                    aspect_category = aspect_sentiment
                    sentiment_label = None  # hoặc 'neutral' nếu dùng tiếng Anh
                
                preprocessed_text = self.preprocess_text(text)
                texts.append(preprocessed_text)
                aspects.append(aspect_category)
                sentiments.append(sentiment_label)
        
        return texts, aspects, sentiments

    
    def prepare_data(self, train_file: str, val_file: str, test_file: str):
        """Chuẩn bị dữ liệu cho training"""
        # Load dữ liệu
        train_data = self.load_json_data(train_file)
        val_data = self.load_json_data(val_file)
        test_data = self.load_json_data(test_file)
        
        # Trích xuất features
        train_texts, train_aspects, train_sentiments = self.extract_aspects_sentiments(train_data)
        val_texts, val_aspects, val_sentiments = self.extract_aspects_sentiments(val_data)
        test_texts, test_aspects, test_sentiments = self.extract_aspects_sentiments(test_data)
        
        # Fit encoders trên tập train
        all_aspects = train_aspects + val_aspects + test_aspects
        all_sentiments = train_sentiments + val_sentiments + test_sentiments
        
        self.aspect_encoder.fit(all_aspects)
        self.sentiment_encoder.fit(all_sentiments)
        
        # Encode labels
        train_aspects_encoded = self.aspect_encoder.transform(train_aspects)
        val_aspects_encoded = self.aspect_encoder.transform(val_aspects)
        test_aspects_encoded = self.aspect_encoder.transform(test_aspects)
        
        train_sentiments_encoded = self.sentiment_encoder.transform(train_sentiments)
        val_sentiments_encoded = self.sentiment_encoder.transform(val_sentiments)
        test_sentiments_encoded = self.sentiment_encoder.transform(test_sentiments)
        
        return {
            'train': (train_texts, train_aspects_encoded, train_sentiments_encoded),
            'val': (val_texts, val_aspects_encoded, val_sentiments_encoded),
            'test': (test_texts, test_aspects_encoded, test_sentiments_encoded)
        }
    
    def save_encoders(self, model_dir: str):
        """Lưu encoders"""
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'aspect_encoder.pkl'), 'wb') as f:
            pickle.dump(self.aspect_encoder, f)
        with open(os.path.join(model_dir, 'sentiment_encoder.pkl'), 'wb') as f:
            pickle.dump(self.sentiment_encoder, f)

class TextVectorizer:
    def __init__(self, max_features=10000, ngram_range=(1, 3), min_df=2, max_df=0.95):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            stop_words=None  # Không sử dụng stop words cho tiếng Việt
        )
    
    def fit_transform(self, texts: List[str]):
        """Fit và transform texts"""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts: List[str]):
        """Transform texts"""
        return self.vectorizer.transform(texts)
    
    def save(self, model_dir: str):
        """Lưu vectorizer"""
        os.makedirs(model_dir, exist_ok=True)
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)