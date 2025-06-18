# trainer.py
import torch
from torch.utils.data import DataLoader as TorchDataLoader
from tqdm import tqdm
import numpy as np
from transformers import (
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import logging
import os
from datetime import datetime
from model import MultiTaskABSAModel, ABSADataset
from utils import MetricsCalculator
from config import ModelConfig, TrainingConfig
from typing import Dict, Tuple

class ABSATrainer:
    def __init__(self, model: MultiTaskABSAModel, config: ModelConfig, 
                 training_config: TrainingConfig):
        self.model = model
        self.config = config
        self.training_config = training_config
        self.metrics_calculator = MetricsCalculator()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        self._setup_logging()
        self.best_f1 = 0.0
        self.patience_counter = 0
        
    def _setup_logging(self):
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
        train_loader = TorchDataLoader(
            train_dataset, batch_size=self.config.batch_size, shuffle=True,
            collate_fn=self._collate_fn
        )
        val_loader = TorchDataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False,
            collate_fn=self._collate_fn
        )
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

        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training phase - returns combined loss and individual losses
            train_combined_loss, train_losses = self._train_epoch(train_loader, optimizer, scheduler)
            
            # Validation phase
            val_combined_loss, val_metrics = self._validate_epoch(val_loader)

            # Log metrics with combined loss
            self._log_epoch_metrics(
                epoch,
                train_combined_loss, train_losses,
                val_combined_loss, val_metrics
            )

            # Early stopping logic
            current_f1 = val_metrics['macro_f1']
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.patience_counter = 0
                self._save_model(
                    "best_model",
                    train_combined_loss, train_losses,
                    val_combined_loss, val_metrics
                )
                self.logger.info(f"New best F1: {current_f1:.4f} - Model saved!")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.training_config.early_stopping_patience:
                    self.logger.info("Early stopping triggered")
                    break

        self.logger.info("Training completed!")

    def _train_epoch(self, train_loader, optimizer, scheduler) -> Tuple[float, Dict[str, float]]:
        """Train for one epoch and return combined loss and individual losses"""
        self.model.train()
        total_combined_loss = 0.0
        total_aspect_loss = 0.0
        total_sentiment_loss = 0.0
        
        for batch in tqdm(train_loader, desc="Training"):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            optimizer.zero_grad()
            
            outputs = self.model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                aspect_labels=batch['aspect_labels'],
                sentiment_labels=batch['sentiment_labels']
            )
            
            # Get the combined loss (sum of aspect and sentiment losses)
            combined_loss = outputs['loss']
            
            # Backpropagate the combined loss
            combined_loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.training_config.max_grad_norm
            )
            optimizer.step()
            scheduler.step()
            
            # Accumulate losses for logging
            total_combined_loss += combined_loss.item()
            total_aspect_loss += outputs['aspect_loss'].item()
            total_sentiment_loss += outputs['sentiment_loss'].item()
        
        # Return average losses
        return (
            total_combined_loss / len(train_loader),
            {
                'aspect_loss': total_aspect_loss / len(train_loader),
                'sentiment_loss': total_sentiment_loss / len(train_loader)
            }
        )

    def _validate_epoch(self, val_loader) -> Tuple[float, Dict[str, float]]:
        """Validate for one epoch and return combined loss and metrics"""
        self.model.eval()
        all_preds = []
        all_labels = []
        total_combined_loss = 0.0
        total_aspect_loss = 0.0
        total_sentiment_loss = 0.0

        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    aspect_labels=batch['aspect_labels'],
                    sentiment_labels=batch['sentiment_labels']
                )
                
                # Get combined loss
                combined_loss = outputs['loss']
                total_combined_loss += combined_loss.item()
                
                # Accumulate individual losses for logging
                total_aspect_loss += outputs['aspect_loss'].item()
                total_sentiment_loss += outputs['sentiment_loss'].item()

                # Get predictions
                aspect_preds = torch.sigmoid(outputs['aspect_logits']) > 0.5
                sentiment_preds = torch.sigmoid(outputs['sentiment_logits']) > 0.5
                
                # Combine predictions and labels
                combined_preds = torch.cat([aspect_preds, sentiment_preds], dim=1)
                combined_labels = torch.cat([batch['aspect_labels'], batch['sentiment_labels']], dim=1)
                
                all_preds.extend(combined_preds.cpu().numpy())
                all_labels.extend(combined_labels.cpu().numpy())

        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        combined_labels_list = self.config.aspect_labels + self.config.sentiment_labels
        
        metrics = self.metrics_calculator.calculate_multi_label_metrics(
            all_labels, all_preds, combined_labels_list
        )
        
        # Add loss information to metrics
        metrics.update({
            'aspect_loss': total_aspect_loss / len(val_loader),
            'sentiment_loss': total_sentiment_loss / len(val_loader),
            'combined_loss': total_combined_loss / len(val_loader)
        })
        
        return (
            total_combined_loss / len(val_loader),
            metrics
        )

    def _log_epoch_metrics(self, epoch: int,
                          train_combined_loss: float, train_losses: Dict[str, float],
                          val_combined_loss: float, val_metrics: Dict[str, float]):
        """Log training and validation metrics for the epoch"""
        self.logger.info(f"\nEpoch {epoch + 1} Metrics:")
        self.logger.info(f"  Training:")
        self.logger.info(f"    Combined Loss: {train_combined_loss:.4f}")
        self.logger.info(f"    Aspect Loss: {train_losses['aspect_loss']:.4f}")
        self.logger.info(f"    Sentiment Loss: {train_losses['sentiment_loss']:.4f}")
        
        self.logger.info(f"  Validation:")
        self.logger.info(f"    Combined Loss: {val_combined_loss:.4f}")
        self.logger.info(f"    Aspect Loss: {val_metrics['aspect_loss']:.4f}")
        self.logger.info(f"    Sentiment Loss: {val_metrics['sentiment_loss']:.4f}")
        self.logger.info(f"    Accuracy: {val_metrics['accuracy']:.4f}")
        self.logger.info(f"    F1 Macro: {val_metrics['macro_f1']:.4f}")
        self.logger.info(f"    F1 Weighted: {val_metrics['weighted_f1']:.4f}")

    def _collate_fn(self, batch):
        return {
            'input_ids': torch.stack([item['input_ids'] for item in batch]),
            'attention_mask': torch.stack([item['attention_mask'] for item in batch]),
            'aspect_labels': torch.stack([item['aspect_labels'] for item in batch]),
            'sentiment_labels': torch.stack([item['sentiment_labels'] for item in batch])
        }

    def _save_model(self, name: str, 
                   train_combined_loss: float, train_losses: Dict[str, float],
                   val_combined_loss: float, val_metrics: Dict[str, float]):
        os.makedirs(self.training_config.model_save_path, exist_ok=True)
        save_path = os.path.join(self.training_config.model_save_path, f"{name}.pt")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'best_f1': self.best_f1,
            'train_combined_loss': train_combined_loss,
            'train_aspect_loss': train_losses['aspect_loss'],
            'train_sentiment_loss': train_losses['sentiment_loss'],
            'val_combined_loss': val_combined_loss,
            'val_aspect_loss': val_metrics['aspect_loss'],
            'val_sentiment_loss': val_metrics['sentiment_loss'],
            **val_metrics
        }
        
        torch.save(save_dict, save_path)
        self.logger.info(f"Model saved to {save_path}")