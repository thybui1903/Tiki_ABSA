import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModel, AutoTokenizer, 
    get_linear_schedule_with_warmup
)
from typing import Dict, List, Tuple, Optional
import numpy as np
from config import ModelConfig
from utils import TextProcessor
from torch.optim import AdamW


class ABSADataset(Dataset):
    """Dataset class for ABSA data"""
    
    def __init__(self, data: List[Dict], data_loader: DataLoader):
        self.data = data
        self.data_loader = data_loader
        self.samples = [self.data_loader.prepare_sample(item) for item in data]
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]

class MultiTaskABSAModel(nn.Module):
    """Multi-task ABSA model with DistilBERT backbone"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # Load pre-trained DistilBERT
        self.bert = AutoModel.from_pretrained(config.model_name)
        
        # Freeze some layers if needed
        # self._freeze_bert_layers(freeze_layers=6)
        
        # Shared representation layer
        self.shared_layer = nn.Sequential(
            nn.Linear(self.bert.config.hidden_size, 512),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate)
        )
        
        # Task-specific heads
        self.aspect_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, len(config.aspect_labels))
        )
        
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(config.dropout_rate),
            nn.Linear(256, len(config.sentiment_labels))
        )
        
        # Loss functions
        self.aspect_criterion = nn.BCEWithLogitsLoss()
        self.sentiment_criterion = nn.BCEWithLogitsLoss()
        
    def forward(self, input_ids, attention_mask, aspect_labels=None, sentiment_labels=None):
        # Get BERT embeddings
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        
        # Use [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Shared representation
        shared_repr = self.shared_layer(cls_output)
        
        # Task-specific predictions
        aspect_logits = self.aspect_classifier(shared_repr)
        sentiment_logits = self.sentiment_classifier(shared_repr)
        
        # Calculate losses if labels are provided
        total_loss = None
        if aspect_labels is not None and sentiment_labels is not None:
            aspect_loss = self.aspect_criterion(aspect_logits, aspect_labels)
            sentiment_loss = self.sentiment_criterion(sentiment_logits, sentiment_labels)
            
            total_loss = (self.config.aspect_weight * aspect_loss + 
                         self.config.sentiment_weight * sentiment_loss)
        
        return {
            'loss': total_loss,
            'aspect_logits': aspect_logits,
            'sentiment_logits': sentiment_logits,
            'aspect_loss': aspect_loss if aspect_labels is not None else None,
            'sentiment_loss': sentiment_loss if sentiment_labels is not None else None
        }
    
    def _freeze_bert_layers(self, freeze_layers: int):
        """Freeze first N layers of BERT"""
        for param in self.bert.embeddings.parameters():
            param.requires_grad = False
            
        for i in range(freeze_layers):
            for param in self.bert.transformer.layer[i].parameters():
                param.requires_grad = False

class ABSAPredictor:
    """Prediction wrapper for the trained model"""
    
    def __init__(self, model: MultiTaskABSAModel, tokenizer, config: ModelConfig):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.text_processor = TextProcessor()
        
        # Set model to evaluation mode
        self.model.eval()
    
    def predict(self, text: str) -> Dict:
        """Predict aspects and sentiments for given text"""
        # Clean text
        cleaned_text = self.text_processor.clean_text(text)
        
        # Tokenize
        encoding = self.tokenizer(
            cleaned_text,
            max_length=self.config.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        with torch.no_grad():
            outputs = self.model(
                input_ids=encoding['input_ids'],
                attention_mask=encoding['attention_mask']
            )
        
        # Apply sigmoid and convert to probabilities
        aspect_probs = torch.sigmoid(outputs['aspect_logits']).cpu().numpy()[0]
        sentiment_probs = torch.sigmoid(outputs['sentiment_logits']).cpu().numpy()[0]
        
        # Get predictions (threshold = 0.5)
        aspect_preds = (aspect_probs > 0.5).astype(int)
        sentiment_preds = (sentiment_probs > 0.5).astype(int)
        
        # Convert to readable format
        predicted_aspects = [
            self.config.aspect_labels[i] for i, pred in enumerate(aspect_preds) if pred == 1
        ]
        predicted_sentiments = [
            self.config.sentiment_labels[i] for i, pred in enumerate(sentiment_preds) if pred == 1
        ]
        
        return {
            'text': cleaned_text,
            'predicted_aspects': predicted_aspects,
            'predicted_sentiments': predicted_sentiments,
            'aspect_probabilities': dict(zip(self.config.aspect_labels, aspect_probs)),
            'sentiment_probabilities': dict(zip(self.config.sentiment_labels, sentiment_probs))
        }
    
    def predict_batch(self, texts: List[str]) -> List[Dict]:
        """Predict for a batch of texts"""
        return [self.predict(text) for text in texts]
