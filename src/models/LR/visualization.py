import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import pandas as pd
from typing import List, Dict, Tuple, Any
import os
from sklearn.metrics import classification_report


plt.style.use('seaborn-v0_8')
plt.rcParams['font.family'] = 'DejaVu Sans'
sns.set_palette("husl")

class ResultVisualizer:
    def __init__(self, results_dir: str):
        self.results_dir = results_dir
        os.makedirs(results_dir, exist_ok=True)
    
    def plot_training_history(self, train_results: Dict, val_results: Dict):
        """Vẽ biểu đồ training history"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Aspect Accuracy
        axes[0,0].bar(['Train', 'Validation'], 
                     [train_results['aspect_accuracy'], val_results['aspect_accuracy']], 
                     color=['#2E86AB', '#A23B72'])
        axes[0,0].set_title('Aspect Detection Accuracy')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim(0, 1)
        
        # Sentiment Accuracy
        axes[0,1].bar(['Train', 'Validation'], 
                     [train_results['sentiment_accuracy'], val_results['sentiment_accuracy']], 
                     color=['#F18F01', '#C73E1D'])
        axes[0,1].set_title('Sentiment Classification Accuracy')
        axes[0,1].set_ylabel('Accuracy')
        axes[0,1].set_ylim(0, 1)
        
        # F1-Scores cho Aspect
        aspect_f1_train = train_results['aspect_report']['weighted avg']['f1-score']
        aspect_f1_val = val_results['aspect_report']['weighted avg']['f1-score']
        axes[1,0].bar(['Train', 'Validation'], [aspect_f1_train, aspect_f1_val], 
                     color=['#2E86AB', '#A23B72'])
        axes[1,0].set_title('Aspect Detection F1-Score')
        axes[1,0].set_ylabel('F1-Score')
        axes[1,0].set_ylim(0, 1)
        
        # F1-Scores cho Sentiment
        sentiment_f1_train = train_results['sentiment_report']['weighted avg']['f1-score']
        sentiment_f1_val = val_results['sentiment_report']['weighted avg']['f1-score']
        axes[1,1].bar(['Train', 'Validation'], [sentiment_f1_train, sentiment_f1_val], 
                     color=['#F18F01', '#C73E1D'])
        axes[1,1].set_title('Sentiment Classification F1-Score')
        axes[1,1].set_ylabel('F1-Score')
        axes[1,1].set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_confusion_matrices(self, y_true_aspect, y_pred_aspect, y_true_sentiment, y_pred_sentiment, 
                               aspect_labels, sentiment_labels):
        """Vẽ confusion matrices"""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Aspect Confusion Matrix
        cm_aspect = confusion_matrix(y_true_aspect, y_pred_aspect)
        sns.heatmap(cm_aspect, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=aspect_labels, yticklabels=aspect_labels, ax=axes[0])
        axes[0].set_title('Aspect Detection Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        
        # Sentiment Confusion Matrix
        cm_sentiment = confusion_matrix(y_true_sentiment, y_pred_sentiment)
        sns.heatmap(cm_sentiment, annot=True, fmt='d', cmap='Oranges', 
                   xticklabels=sentiment_labels, yticklabels=sentiment_labels, ax=axes[1])
        axes[1].set_title('Sentiment Classification Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrices.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_classification_report(self, aspect_report, sentiment_report, aspect_labels, sentiment_labels):
        """Vẽ classification report"""
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        metrics = ['precision', 'recall', 'f1-score']
        width = 0.25

        # ==== ASPECT ====
        aspect_df = pd.DataFrame(aspect_report).transpose()
        present_aspect_labels = [label for label in aspect_labels if label in aspect_df.index]
        aspect_df = aspect_df.loc[present_aspect_labels]
        x = np.arange(len(present_aspect_labels))

        for i, metric in enumerate(metrics):
            if metric in aspect_df.columns and len(aspect_df[metric]) == len(x):
                axes[0].bar(x + i * width, aspect_df[metric], width, label=metric.capitalize())
            else:
                print(f"[WARNING] Aspect metric '{metric}' skipped due to shape mismatch or missing.")

        axes[0].set_xlabel('Aspect Categories')
        axes[0].set_ylabel('Score')
        axes[0].set_title('Aspect Detection Performance by Category')
        axes[0].set_xticks(x + width)
        axes[0].set_xticklabels(present_aspect_labels, rotation=45)
        axes[0].legend()
        axes[0].set_ylim(0, 1)

        # ==== SENTIMENT ====
        sentiment_df = pd.DataFrame(sentiment_report).transpose()
        present_sentiment_labels = [label for label in sentiment_labels if label in sentiment_df.index]
        sentiment_df = sentiment_df.loc[present_sentiment_labels]
        x = np.arange(len(present_sentiment_labels))

        for i, metric in enumerate(metrics):
            if metric in sentiment_df.columns and len(sentiment_df[metric]) == len(x):
                axes[1].bar(x + i * width, sentiment_df[metric], width, label=metric.capitalize())
            else:
                print(f"[WARNING] Sentiment metric '{metric}' skipped due to shape mismatch or missing.")

        axes[1].set_xlabel('Sentiment Categories')
        axes[1].set_ylabel('Score')
        axes[1].set_title('Sentiment Classification Performance by Category')
        axes[1].set_xticks(x + width)
        axes[1].set_xticklabels(present_sentiment_labels, rotation=45)
        axes[1].legend()
        axes[1].set_ylim(0, 1)

        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'classification_reports.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_error_analysis(self, errors_df):
        """Phân tích lỗi"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Phân bố lỗi theo aspect
        aspect_errors = errors_df['true_aspect'].value_counts()
        axes[0,0].pie(aspect_errors.values, labels=aspect_errors.index, autopct='%1.1f%%')
        axes[0,0].set_title('Distribution of Aspect Prediction Errors')
        
        # Phân bố lỗi theo sentiment
        sentiment_errors = errors_df['true_sentiment'].value_counts()
        axes[0,1].pie(sentiment_errors.values, labels=sentiment_errors.index, autopct='%1.1f%%')
        axes[0,1].set_title('Distribution of Sentiment Prediction Errors')
        
        # Top words trong các câu bị lỗi
        if 'text_length' in errors_df.columns:
            axes[1,0].hist(errors_df['text_length'], bins=20, alpha=0.7, color='red')
            axes[1,0].set_title('Text Length Distribution in Errors')
            axes[1,0].set_xlabel('Text Length')
            axes[1,0].set_ylabel('Frequency')
        
        # Error rate theo category
        error_rate_aspect = errors_df.groupby('true_aspect').size()
        axes[1,1].bar(error_rate_aspect.index, error_rate_aspect.values, color='coral')
        axes[1,1].set_title('Error Count by Aspect Category')
        axes[1,1].set_xlabel('Aspect Category')
        axes[1,1].set_ylabel('Error Count')
        axes[1,1].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()