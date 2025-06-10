# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report
from typing import List, Dict, Tuple, Optional
import matplotlib.font_manager as fm
from itertools import cycle
import warnings
import os
from torch.utils.data import DataLoader as TorchDataLoader
from torch.optim import AdamW
import torch
import torch.nn as nn
from tqdm import tqdm
from trainer import ABSATrainer
from transformers import get_linear_schedule_with_warmup
warnings.filterwarnings('ignore')

# Set Vietnamese font support (optional)
plt.rcParams['font.family'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

class ABSAVisualizer:
    """Visualization utilities for ABSA model"""
    
    def __init__(self, figsize: Tuple[int, int] = (12, 8)):
        self.figsize = figsize
        self.colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD', '#98D8C8']
        
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                            labels: List[str], task_name: str = "Classification",
                            normalize: bool = True, save_path: Optional[str] = None):
        """
        Plot confusion matrix for multi-label classification
        """
        n_classes = len(labels)
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'{task_name} - Confusion Matrices', fontsize=16, fontweight='bold')
        
        # Flatten axes for easier indexing
        axes = axes.flatten()
        
        for i, label in enumerate(labels):
            if i >= len(axes):
                break
                
            # Extract binary classification for each class
            y_true_binary = y_true[:, i]
            y_pred_binary = y_pred[:, i]
            
            # Compute confusion matrix
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            
            if normalize:
                cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
                fmt = '.2f'
                title = f'{label}\n(Normalized)'
            else:
                fmt = 'd'
                title = f'{label}\n(Raw Counts)'
            
            # Plot heatmap
            sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=axes[i], cbar_kws={'shrink': 0.8})
            
            axes[i].set_title(title, fontweight='bold')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        # Hide unused subplots
        for j in range(len(labels), len(axes)):
            axes[j].axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_training_history(self, history: Dict[str, List[float]], 
                            save_path: Optional[str] = None):
        """
        Plot training and validation loss/metrics over epochs
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training History', fontsize=16, fontweight='bold')
        
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Plot 1: Loss curves
        axes[0, 0].plot(epochs, history['train_loss'], 'b-o', label='Training Loss', linewidth=2)
        if 'val_loss' in history:
            axes[0, 0].plot(epochs, history['val_loss'], 'r-s', label='Validation Loss', linewidth=2)
        axes[0, 0].set_title('Loss Curves', fontweight='bold')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Aspect F1 scores
        if 'train_aspect_f1' in history:
            axes[0, 1].plot(epochs, history['train_aspect_f1'], 'g-o', label='Train Aspect F1', linewidth=2)
        if 'val_aspect_f1' in history:
            axes[0, 1].plot(epochs, history['val_aspect_f1'], 'm-s', label='Val Aspect F1', linewidth=2)
        axes[0, 1].set_title('Aspect F1 Scores', fontweight='bold')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('F1 Score')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Sentiment F1 scores
        if 'train_sentiment_f1' in history:
            axes[1, 0].plot(epochs, history['train_sentiment_f1'], 'c-o', label='Train Sentiment F1', linewidth=2)
        if 'val_sentiment_f1' in history:
            axes[1, 0].plot(epochs, history['val_sentiment_f1'], 'orange', marker='s', label='Val Sentiment F1', linewidth=2)
        axes[1, 0].set_title('Sentiment F1 Scores', fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('F1 Score')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Plot 4: Combined metrics
        if 'val_aspect_f1' in history and 'val_sentiment_f1' in history:
            combined_f1 = [(a + s) / 2 for a, s in zip(history['val_aspect_f1'], history['val_sentiment_f1'])]
            axes[1, 1].plot(epochs, combined_f1, 'purple', marker='d', label='Combined F1', linewidth=2)
        axes[1, 1].set_title('Combined Validation F1', fontweight='bold')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('F1 Score')
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_class_performance(self, metrics_dict: Dict[str, Dict[str, float]], 
                             task_name: str = "Classification",
                             save_path: Optional[str] = None):
        """
        Plot per-class performance metrics (Precision, Recall, F1)
        """
        classes = list(metrics_dict.keys())
        metrics = ['precision', 'recall', 'f1-score']
        
        # Prepare data
        data = {metric: [] for metric in metrics}
        for class_name in classes:
            for metric in metrics:
                data[metric].append(metrics_dict[class_name].get(metric, 0))
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(classes))
        width = 0.25
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        for i, (metric, values) in enumerate(data.items()):
            bars = ax.bar(x + i * width, values, width, label=metric.capitalize(), 
                         color=colors[i], alpha=0.8)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Classes', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title(f'{task_name} - Per-Class Performance', fontweight='bold', fontsize=14)
        ax.set_xticks(x + width)
        ax.set_xticklabels(classes, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_prediction_distribution(self, predictions: List[Dict], 
                                   save_path: Optional[str] = None):
        """
        Plot distribution of predictions (aspects and sentiments)
        """
        # Extract prediction data
        aspect_counts = {}
        sentiment_counts = {}
        
        for pred in predictions:
            for aspect in pred.get('predicted_aspects', []):
                aspect_counts[aspect] = aspect_counts.get(aspect, 0) + 1
            for sentiment in pred.get('predicted_sentiments', []):
                sentiment_counts[sentiment] = sentiment_counts.get(sentiment, 0) + 1
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Aspect distribution
        if aspect_counts:
            aspects = list(aspect_counts.keys())
            counts = list(aspect_counts.values())
            
            bars1 = ax1.bar(aspects, counts, color=self.colors[:len(aspects)], alpha=0.8)
            ax1.set_title('Aspect Distribution', fontweight='bold', fontsize=14)
            ax1.set_xlabel('Aspects', fontweight='bold')
            ax1.set_ylabel('Frequency', fontweight='bold')
            ax1.tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar in bars1:
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        # Sentiment distribution
        if sentiment_counts:
            sentiments = list(sentiment_counts.keys())
            counts = list(sentiment_counts.values())
            
            bars2 = ax2.bar(sentiments, counts, color=self.colors[3:3+len(sentiments)], alpha=0.8)
            ax2.set_title('Sentiment Distribution', fontweight='bold', fontsize=14)
            ax2.set_xlabel('Sentiments', fontweight='bold')
            ax2.set_ylabel('Frequency', fontweight='bold')
            
            # Add value labels
            for bar in bars2:
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{int(height)}', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_learning_curves(self, train_sizes: List[int], train_scores: List[float], 
                           val_scores: List[float], metric_name: str = "F1 Score",
                           save_path: Optional[str] = None):
        """
        Plot learning curves to show model performance vs training data size
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Calculate mean and std
        train_mean = np.mean(train_scores, axis=1) if len(np.array(train_scores).shape) > 1 else train_scores
        val_mean = np.mean(val_scores, axis=1) if len(np.array(val_scores).shape) > 1 else val_scores
        
        # Plot curves
        ax.plot(train_sizes, train_mean, 'o-', color='#FF6B6B', linewidth=2, 
                label=f'Training {metric_name}')
        ax.plot(train_sizes, val_mean, 'o-', color='#4ECDC4', linewidth=2, 
                label=f'Validation {metric_name}')
        
        # Fill between if std is available
        if len(np.array(train_scores).shape) > 1:
            train_std = np.std(train_scores, axis=1)
            val_std = np.std(val_scores, axis=1)
            
            ax.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, 
                           alpha=0.2, color='#FF6B6B')
            ax.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, 
                           alpha=0.2, color='#4ECDC4')
        
        ax.set_xlabel('Training Set Size', fontweight='bold')
        ax.set_ylabel(metric_name, fontweight='bold')
        ax.set_title(f'Learning Curves - {metric_name}', fontweight='bold', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_model_comparison(self, models_performance: Dict[str, Dict[str, float]],
                            save_path: Optional[str] = None):
        """
        Compare performance of different models
        """
        models = list(models_performance.keys())
        metrics = list(next(iter(models_performance.values())).keys())
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        x = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            values = [models_performance[model][metric] for metric in metrics]
            bars = ax.bar(x + i * width, values, width, label=model, 
                         color=self.colors[i % len(self.colors)], alpha=0.8)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
        
        ax.set_xlabel('Metrics', fontweight='bold')
        ax.set_ylabel('Score', fontweight='bold')
        ax.set_title('Model Performance Comparison', fontweight='bold', fontsize=14)
        ax.set_xticks(x + width * (len(models) - 1) / 2)
        ax.set_xticklabels(metrics, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_attention_heatmap(self, tokens: List[str], attention_weights: np.ndarray,
                             save_path: Optional[str] = None):
        """
        Plot attention weights heatmap
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create heatmap
        im = ax.imshow(attention_weights, cmap='Blues', aspect='auto')
        
        # Set ticks and labels
        ax.set_xticks(range(len(tokens)))
        ax.set_yticks(range(len(tokens)))
        ax.set_xticklabels(tokens, rotation=45, ha='right')
        ax.set_yticklabels(tokens)
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Attention Weight', fontweight='bold')
        
        # Add text annotations
        for i in range(len(tokens)):
            for j in range(len(tokens)):
                text = ax.text(j, i, f'{attention_weights[i, j]:.2f}',
                             ha="center", va="center", color="black" if attention_weights[i, j] < 0.5 else "white")
        
        ax.set_title("Attention Weights Heatmap", fontweight='bold', fontsize=14)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()

# Enhanced trainer with visualization support
class VisualizationTrainer(ABSATrainer):
    """Extended trainer with visualization capabilities"""
    
    def __init__(self, model, config, training_config):
        super().__init__(model, config, training_config)
        self.visualizer = ABSAVisualizer()
        self.training_history = {
            'train_loss': [],
            'val_aspect_f1': [],
            'val_sentiment_f1': [],
            'val_loss': []
        }
    
    def train(self, train_dataset, val_dataset):
        """Enhanced training with history tracking"""
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
        
        for epoch in range(self.config.num_epochs):
            self.logger.info(f"Epoch {epoch + 1}/{self.config.num_epochs}")
            
            # Training phase
            train_loss = self._train_epoch(train_loader, optimizer, scheduler)
            self.training_history['train_loss'].append(train_loss)
            
            # Validation phase
            val_metrics = self._validate_epoch(val_loader)
            self.training_history['val_aspect_f1'].append(val_metrics['aspect_macro_f1'])
            self.training_history['val_sentiment_f1'].append(val_metrics['sentiment_macro_f1'])
            
            self.logger.info(f"Train Loss: {train_loss:.4f}")
            self.logger.info(f"Val Aspect F1: {val_metrics['aspect_macro_f1']:.4f}")
            self.logger.info(f"Val Sentiment F1: {val_metrics['sentiment_macro_f1']:.4f}")
            
            # Check for improvement and early stopping
            current_f1 = (val_metrics['aspect_macro_f1'] + val_metrics['sentiment_macro_f1']) / 2
            
            if current_f1 > self.best_f1:
                self.best_f1 = current_f1
                self.patience_counter = 0
                self._save_model("best_model")
                self.logger.info(f"New best F1: {current_f1:.4f}")
            else:
                self.patience_counter += 1
            
            if self.patience_counter >= self.training_config.early_stopping_patience:
                self.logger.info("Early stopping triggered")
                break
        
        # Plot training history
        self.visualizer.plot_training_history(
            self.training_history, 
            save_path=os.path.join(self.training_config.output_dir, "training_history.png")
        )
        
        self.logger.info("Training completed!")
    
    def evaluate_with_visualization(self, test_dataset, save_dir: str = "C:/Users/DELL/Tiki_ABSA/src/models/DistilBert_MultiTasking/visualizations"):
        """Comprehensive evaluation with visualizations"""
        os.makedirs(save_dir, exist_ok=True)
        
        test_loader = TorchDataLoader(
            test_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn
        )
        
        self.model.eval()
        
        all_aspect_preds = []
        all_aspect_labels = []
        all_sentiment_preds = []
        all_sentiment_labels = []
        predictions = []
        
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask']
                )
                
                aspect_preds = (torch.sigmoid(outputs['aspect_logits']) > 0.5).cpu().numpy()
                sentiment_preds = (torch.sigmoid(outputs['sentiment_logits']) > 0.5).cpu().numpy()
                
                all_aspect_preds.extend(aspect_preds)
                all_aspect_labels.extend(batch['aspect_labels'].cpu().numpy())
                all_sentiment_preds.extend(sentiment_preds)
                all_sentiment_labels.extend(batch['sentiment_labels'].cpu().numpy())
                
                # Store predictions for distribution analysis
                for i in range(len(aspect_preds)):
                    pred_aspects = [self.config.aspect_labels[j] for j, pred in enumerate(aspect_preds[i]) if pred == 1]
                    pred_sentiments = [self.config.sentiment_labels[j] for j, pred in enumerate(sentiment_preds[i]) if pred == 1]
                    
                    predictions.append({
                        'predicted_aspects': pred_aspects,
                        'predicted_sentiments': pred_sentiments
                    })
        
        # Convert to arrays
        all_aspect_preds = np.array(all_aspect_preds)
        all_aspect_labels = np.array(all_aspect_labels)
        all_sentiment_preds = np.array(all_sentiment_preds)
        all_sentiment_labels = np.array(all_sentiment_labels)
        
        # Plot confusion matrices
        self.visualizer.plot_confusion_matrix(
            all_aspect_labels, all_aspect_preds, self.config.aspect_labels,
            task_name="Aspect Classification",
            save_path=os.path.join(save_dir, "aspect_confusion_matrix.png")
        )
        
        self.visualizer.plot_confusion_matrix(
            all_sentiment_labels, all_sentiment_preds, self.config.sentiment_labels,
            task_name="Sentiment Classification", 
            save_path=os.path.join(save_dir, "sentiment_confusion_matrix.png")
        )
        
        # Calculate detailed metrics
        aspect_report = classification_report(all_aspect_labels, all_aspect_preds, 
                                            target_names=self.config.aspect_labels, 
                                            output_dict=True, zero_division=0)
        sentiment_report = classification_report(all_sentiment_labels, all_sentiment_preds,
                                               target_names=self.config.sentiment_labels,
                                               output_dict=True, zero_division=0)
        
        # Plot per-class performance
        aspect_metrics = {label: aspect_report[label] for label in self.config.aspect_labels if label in aspect_report}
        sentiment_metrics = {label: sentiment_report[label] for label in self.config.sentiment_labels if label in sentiment_report}
        
        self.visualizer.plot_class_performance(
            aspect_metrics, "Aspect Classification",
            save_path=os.path.join(save_dir, "aspect_performance.png")
        )
        
        self.visualizer.plot_class_performance(
            sentiment_metrics, "Sentiment Classification",
            save_path=os.path.join(save_dir, "sentiment_performance.png")
        )
        
        # Plot prediction distributions
        self.visualizer.plot_prediction_distribution(
            predictions,
            save_path=os.path.join(save_dir, "prediction_distribution.png")
        )
        
        return {
            'aspect_metrics': aspect_metrics,
            'sentiment_metrics': sentiment_metrics,
            'predictions': predictions
        }

# Example usage
def example_usage():
    """Example of how to use the visualization module"""
    
    # Initialize visualizer
    visualizer = ABSAVisualizer()
    
    # Example 1: Plot training history
    history = {
    'train_loss': [
        1.2101, 0.8785, 0.7414, 0.6195, 0.5302,
        0.4585, 0.4036, 0.3560, 0.3125, 0.2766,
        0.2415, 0.2104, 0.1840, 0.1642
    ],
    'val_aspect_f1': [
        0.4291, 0.4629, 0.4775, 0.6735, 0.6829,
        0.6872, 0.7335, 0.7434, 0.7597, 0.7822,
        0.7886, 0.7656, 0.7737, 0.7733
    ],
    'val_sentiment_f1': [
        0.3166, 0.4433, 0.4554, 0.5154, 0.5346,
        0.5337, 0.5585, 0.5478, 0.5359, 0.5665,
        0.6368, 0.5948, 0.6090, 0.6402
    ],
    'val_loss': [  # Không có val_loss trong log
    ]
    }
    
    visualizer.plot_training_history(history, save_path="training_history.png")
    
    """
    # Example 2: Model comparison
    models_performance = {
        'DistilBERT': {'Micro F1': 0.85, 'Macro F1': 0.78, 'Weighted F1': 0.82},
        'BERT-base': {'Micro F1': 0.87, 'Macro F1': 0.80, 'Weighted F1': 0.84},
        'PhoBERT': {'Micro F1': 0.89, 'Macro F1': 0.83, 'Weighted F1': 0.86}
    }
    
    visualizer.plot_model_comparison(models_performance, save_path="model_comparison.png")
    """
    print("Visualization examples completed!")

if __name__ == "__main__":
    example_usage()