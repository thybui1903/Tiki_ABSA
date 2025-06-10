# trainer.py
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score
from transformers import (
    AutoModel, AutoTokenizer, 
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import logging
import os
from datetime import datetime
from model import MultiTaskABSAModel, ABSADataset
from utils import MetricsCalculator
from config import ModelConfig, TrainingConfig
from typing import Dict
from datetime import datetime

class ABSATrainer:
    """Trainer class for the ABSA model"""
    
    def __init__(self, model: MultiTaskABSAModel, config: ModelConfig, 
                 training_config: TrainingConfig):
        self.model = model
        self.config = config
        self.training_config = training_config
        self.metrics_calculator = MetricsCalculator()
        
        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Setup logging
        self._setup_logging()
        
        # Initialize best metrics
        self.best_f1 = 0.0
        self.patience_counter = 0
        
    def _setup_logging(self):
        """Setup logging configuration"""
        os.makedirs(self.training_config.log_dir, exist_ok=True)
        
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(
                    os.path.join(self.training_config.log_dir, 
                               f'training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
                ),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def train(self, train_dataset: ABSADataset, val_dataset: ABSADataset):
        """Train the model"""
        # Create data loaders
        train_loader = TorchDataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True,
            collate_fn=self._collate_fn
        )
        
        val_loader = TorchDataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        
        total_steps = len(train_loader) * self.config.num_epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.config.warmup_steps,
            num_training_steps=total_steps
        )
        
        self.logger.info(f"Starting training for {self.config.num_epochs} epochs")
        self.logger.info(f"Total training steps: {total_steps}")
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            
            # Validation phase
            val_loss, val_metrics = self._validate_epoch(val_loader)
            
            # Log main metrics
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Loss: {val_loss:.4f}")
            self.logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
            self.logger.info(f"Val F1 Macro: {val_metrics['macro_f1']:.4f}")
            self.logger.info(f"Val F1 Weighted: {val_metrics['weighted_f1']:.4f}")
            
            # Check for improvement using combined F1 score
            current_f1 = val_metrics['macro_f1']
            
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.patience_counter = 0
                self._save_model("best_model")
                self.logger.info(f"New best F1: {current_f1:.4f} - Model saved!")
            else:
                self.patience_counter += 1
                self.logger.info(f"No improvement. Patience: {self.patience_counter}/{self.training_config.early_stopping_patience}")
            
            # Early stopping
            if self.patience_counter >= self.training_config.early_stopping_patience:
                self.logger.info("Early stopping triggered")
                break
        
        self.logger.info("Training completed!")
    
    def _train_epoch(self, train_loader, optimizer, scheduler):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch in progress_bar:
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                aspect_labels=batch['aspect_labels'],
                sentiment_labels=batch['sentiment_labels']
            )
            
            loss = outputs['loss']
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 
                                         self.training_config.max_grad_norm)
            
            optimizer.step()
            scheduler.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
        
        return total_loss / len(train_loader)
    
    def _validate_epoch(self, val_loader):
        """Validate for one epoch"""
        self.model.eval()
        
        all_preds = []
        all_labels = []
        total_loss = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    aspect_labels=batch['aspect_labels'],
                    sentiment_labels=batch['sentiment_labels']
                )
                
                # Accumulate total loss
                total_loss += outputs['loss'].item()
                
                # Combine predictions and labels for overall evaluation
                aspect_preds = torch.sigmoid(outputs['aspect_logits']) > 0.5
                sentiment_preds = torch.sigmoid(outputs['sentiment_logits']) > 0.5
                
                # Concatenate aspect and sentiment predictions/labels
                combined_preds = torch.cat([aspect_preds, sentiment_preds], dim=1)
                combined_labels = torch.cat([batch['aspect_labels'], batch['sentiment_labels']], dim=1)
                
                all_preds.extend(combined_preds.cpu().numpy())
                all_labels.extend(combined_labels.cpu().numpy())
        
        # Calculate overall metrics
        avg_loss = total_loss / len(val_loader)
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        
        # Calculate combined metrics
        combined_labels_list = self.config.aspect_labels + self.config.sentiment_labels
        metrics = self.metrics_calculator.calculate_multi_label_metrics(
            all_labels, all_preds, combined_labels_list
        )
        
        return avg_loss, metrics
    
    def _collate_fn(self, batch):
        """Custom collate function for batching"""
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'aspect_labels': torch.stack([item['aspect_labels'] for item in batch]),
            'sentiment_labels': torch.stack([item['sentiment_labels'] for item in batch])
        }
    
    def _save_model(self, name: str):
        """Save model checkpoint"""
        os.makedirs(self.training_config.model_save_path, exist_ok=True)
        
        save_path = os.path.join(self.training_config.model_save_path, f"{name}.pt")
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_f1': self.best_f1
        }, save_path)
        
        self.logger.info(f"Best model saved to {save_path}")
