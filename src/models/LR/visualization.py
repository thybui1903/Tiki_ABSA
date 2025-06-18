# visualization.py - Fixed for soft confusion matrix display
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
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
    
    def plot_combined_training_history(self, train_results: Dict, val_results: Dict):
        """Vẽ biểu đồ training history cho combined labels"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Overall Accuracy
        axes[0,0].bar(['Train', 'Validation'], 
                     [train_results['accuracy'], val_results['accuracy']], 
                     color=['#2E86AB', '#A23B72'])
        axes[0,0].set_title('Overall Classification Accuracy')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim(0, 1)
        
        # F1-Scores
        train_f1 = train_results['f1_weighted']
        val_f1 = val_results['f1_weighted']
        axes[0,1].bar(['Train', 'Validation'], [train_f1, val_f1], 
                     color=['#F18F01', '#C73E1D'])
        axes[0,1].set_title('Weighted F1-Score')
        axes[0,1].set_ylabel('F1-Score')
        axes[0,1].set_ylim(0, 1)
        
        # Aspect and Sentiment breakdown if available
        if train_results.get('breakdown') and val_results.get('breakdown'):
            # Aspect Accuracy
            axes[1,0].bar(['Train', 'Validation'], 
                         [train_results['breakdown']['aspect_accuracy'], 
                          val_results['breakdown']['aspect_accuracy']], 
                         color=['#2E86AB', '#A23B72'])
            axes[1,0].set_title('Aspect Detection Accuracy')
            axes[1,0].set_ylabel('Accuracy')
            axes[1,0].set_ylim(0, 1)
            
            # Sentiment Accuracy
            axes[1,1].bar(['Train', 'Validation'], 
                         [train_results['breakdown']['sentiment_accuracy'], 
                          val_results['breakdown']['sentiment_accuracy']], 
                         color=['#F18F01', '#C73E1D'])
            axes[1,1].set_title('Sentiment Classification Accuracy')
            axes[1,1].set_ylabel('Accuracy')
            axes[1,1].set_ylim(0, 1)
        else:
            # Macro vs Weighted F1
            axes[1,0].bar(['Train', 'Validation'], 
                         [train_results['f1_macro'], val_results['f1_macro']], 
                         color=['#2E86AB', '#A23B72'])
            axes[1,0].set_title('Macro F1-Score')
            axes[1,0].set_ylabel('F1-Score')
            axes[1,0].set_ylim(0, 1)
            
            # Loss
            axes[1,1].bar(['Train', 'Validation'], 
                         [train_results['loss'], val_results['loss']], 
                         color=['#F18F01', '#C73E1D'])
            axes[1,1].set_title('Log Loss')
            axes[1,1].set_ylabel('Loss')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'training_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    @staticmethod
    def build_soft_confusion_labels(ground_truth_data, prediction_data):
        """
        Sinh danh sách y_true và y_pred theo kiểu soft matching:
        - Đúng nếu nhãn dự đoán có trong nhãn thật.
        - Gán 'None' cho phần thiếu hoặc thừa.
        """
        y_true = []
        y_pred = []

        for gt_sample, pred_sample in zip(ground_truth_data, prediction_data):
            gt_labels = [label[3] for label in gt_sample["labels"]]
            pred_labels = [label[3] for label in pred_sample["labels"]]
            temp_pred_labels = pred_labels.copy()

            for gt_label in gt_labels:
                y_true.append(gt_label)
                if gt_label in temp_pred_labels:
                    y_pred.append(gt_label)
                    temp_pred_labels.remove(gt_label)
                else:
                    y_pred.append("None")  # thiếu

            for extra_pred in temp_pred_labels:
                y_true.append("None")  # thừa
                y_pred.append(extra_pred)

        return y_true, y_pred

    def plot_soft_confusion_matrix(self, y_true, y_pred, figsize=(10, 8), save_path=None):
        """
        Vẽ confusion matrix từ y_true và y_pred đã xử lý theo soft matching.
        """
        all_labels = sorted(set(y_true + y_pred))
        cm = confusion_matrix(y_true, y_pred, labels=all_labels)

        fig, ax = plt.subplots(figsize=figsize)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=all_labels)
        disp.plot(ax=ax, xticks_rotation=45, cmap='Blues')
        plt.title("Confusion Matrix (Soft Matching)")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_combined_confusion_matrix(self, y_true, y_pred, label_names, label_encoder=None):
        """Plot confusion matrix for combined labels"""
        # Convert numeric labels to names if encoder is provided
        if label_encoder is not None:
            y_true_names = label_encoder.inverse_transform(y_true)
            y_pred_names = label_encoder.inverse_transform(y_pred)
        else:
            y_true_names = y_true
            y_pred_names = y_pred
        
        # Create confusion matrix
        unique_labels = sorted(set(list(y_true_names) + list(y_pred_names)))
        
        # Limit the number of labels for readability
        if len(unique_labels) > 20:
            # Only show most frequent labels
            from collections import Counter
            label_counts = Counter(y_true_names)
            most_common_labels = [label for label, _ in label_counts.most_common(20)]
            
            # Filter data to only include most common labels
            filtered_true = []
            filtered_pred = []
            for true_label, pred_label in zip(y_true_names, y_pred_names):
                if true_label in most_common_labels or pred_label in most_common_labels:
                    filtered_true.append(true_label if true_label in most_common_labels else 'Other')
                    filtered_pred.append(pred_label if pred_label in most_common_labels else 'Other')
            
            unique_labels = most_common_labels + ['Other']
            y_true_names = filtered_true
            y_pred_names = filtered_pred
        
        cm = confusion_matrix(y_true_names, y_pred_names, labels=unique_labels)
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(max(10, len(unique_labels) * 0.5), max(8, len(unique_labels) * 0.4)))
        
        # Truncate labels for display
        display_labels = [str(label)[:10] + '...' if len(str(label)) > 10 else str(label) for label in unique_labels]
        
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=display_labels)
        disp.plot(ax=ax, xticks_rotation=45, cmap='Blues', values_format='d')
        
        plt.title('Confusion Matrix (Combined Labels)')
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
    def plot_combined_classification_report(self, classification_report_dict, label_names):
        """Vẽ classification report cho combined labels"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Convert classification report to DataFrame
        df = pd.DataFrame(classification_report_dict).transpose()
        
        # Remove the summary rows for now
        df = df.drop(['accuracy', 'macro avg', 'weighted avg'], errors='ignore')
        
        # Only keep metrics we care about
        metrics = ['precision', 'recall', 'f1-score']
        df = df[metrics]
        
        # Sort by f1-score and handle all labels
        df = df.sort_values('f1-score', ascending=False)
        
        # If there are too many labels, create multiple plots or use a different visualization
        if len(df) > 20:
            # Create a heatmap instead of bar plot for many labels
            fig, ax = plt.subplots(1, 1, figsize=(8, max(10, len(df) * 0.3)))
            
            # Create heatmap
            sns.heatmap(df.values, annot=True, fmt='.3f', cmap='YlOrRd', 
                    xticklabels=metrics, yticklabels=df.index, ax=ax)
            ax.set_title('Classification Performance Heatmap')
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Combined Labels')
            
            # Rotate y-axis labels for better readability
            plt.yticks(rotation=0, fontsize=8)
            
        else:
            # Create bar plot for fewer labels
            x = np.arange(len(df))
            width = 0.25
            
            for i, metric in enumerate(metrics):
                ax.bar(x + i * width, df[metric], width, label=metric.capitalize())
            
            # Truncate labels for display
            display_labels = [str(idx)[:15] + '...' if len(str(idx)) > 15 else str(idx) for idx in df.index]
            
            ax.set_xlabel('Combined Labels')
            ax.set_ylabel('Score')
            ax.set_title('Classification Performance by Combined Label')
            ax.set_xticks(x + width)
            ax.set_xticklabels(display_labels, rotation=45, ha='right')
            ax.legend()
            ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'classification_report.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_combined_comprehensive_metrics(self, train_results, val_results, test_results):
        """Vẽ biểu đồ tổng hợp các metrics"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        datasets = ['Train', 'Validation', 'Test']
        
        # Accuracy
        accuracies = [train_results['accuracy'], val_results['accuracy'], test_results['accuracy']]
        axes[0,0].bar(datasets, accuracies, color=['#2E86AB', '#A23B72', '#F18F01'])
        axes[0,0].set_title('Accuracy Across Datasets')
        axes[0,0].set_ylabel('Accuracy')
        axes[0,0].set_ylim(0, 1)
        
        # F1 Scores
        f1_macro = [train_results['f1_macro'], val_results['f1_macro'], test_results['f1_macro']]
        f1_weighted = [train_results['f1_weighted'], val_results['f1_weighted'], test_results['f1_weighted']]
        
        x = np.arange(len(datasets))
        width = 0.35
        
        axes[0,1].bar(x - width/2, f1_macro, width, label='Macro F1', color='#2E86AB')
        axes[0,1].bar(x + width/2, f1_weighted, width, label='Weighted F1', color='#A23B72')
        axes[0,1].set_title('F1-Scores Comparison')
        axes[0,1].set_ylabel('F1-Score')
        axes[0,1].set_xticks(x)
        axes[0,1].set_xticklabels(datasets)
        axes[0,1].legend()
        axes[0,1].set_ylim(0, 1)
        
        # Loss
        losses = [train_results['loss'], val_results['loss'], test_results['loss']]
        axes[1,0].bar(datasets, losses, color=['#F18F01', '#C73E1D', '#91C7B1'])
        axes[1,0].set_title('Loss Across Datasets')
        axes[1,0].set_ylabel('Log Loss')
        
        # Breakdown metrics if available
        if all(results.get('breakdown') for results in [train_results, val_results, test_results]):
            aspect_acc = [results['breakdown']['aspect_accuracy'] for results in [train_results, val_results, test_results]]
            sentiment_acc = [results['breakdown']['sentiment_accuracy'] for results in [train_results, val_results, test_results]]
            
            x = np.arange(len(datasets))
            axes[1,1].bar(x - width/2, aspect_acc, width, label='Aspect', color='#2E86AB')
            axes[1,1].bar(x + width/2, sentiment_acc, width, label='Sentiment', color='#F18F01')
            axes[1,1].set_title('Aspect vs Sentiment Accuracy')
            axes[1,1].set_ylabel('Accuracy')
            axes[1,1].set_xticks(x)
            axes[1,1].set_xticklabels(datasets)
            axes[1,1].legend()
            axes[1,1].set_ylim(0, 1)
        else:
            # Show class distribution instead
            if 'support' in test_results and test_results['support'] is not None:
                support = test_results['support']
                # Only show non-zero support
                non_zero_support = support[support > 0]
                if len(non_zero_support) > 0:
                    axes[1,1].pie(non_zero_support, autopct='%1.1f%%', startangle=90)
                    axes[1,1].set_title('Class Distribution in Test Set (Non-zero only)')
                else:
                    axes[1,1].text(0.5, 0.5, 'No class distribution\ndata available', 
                                  horizontalalignment='center', verticalalignment='center',
                                  transform=axes[1,1].transAxes, fontsize=12)
                    axes[1,1].set_title('Class Distribution')
            else:
                axes[1,1].text(0.5, 0.5, 'Breakdown metrics\nnot available', 
                              horizontalalignment='center', verticalalignment='center',
                              transform=axes[1,1].transAxes, fontsize=12)
                axes[1,1].set_title('Additional Metrics')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'comprehensive_metrics.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_combined_error_analysis(self, errors_df):
        """Phân tích lỗi cho combined labels"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Distribution of errors by true aspect
        if 'true_aspect' in errors_df.columns:
            aspect_errors = errors_df['true_aspect'].value_counts().head(10)
            axes[0,0].pie(aspect_errors.values, labels=aspect_errors.index, autopct='%1.1f%%')
            axes[0,0].set_title('Error Distribution by True Aspect')
        
        # Distribution of errors by true sentiment
        if 'true_sentiment' in errors_df.columns:
            sentiment_errors = errors_df['true_sentiment'].value_counts()
            axes[0,1].pie(sentiment_errors.values, labels=sentiment_errors.index, autopct='%1.1f%%')
            axes[0,1].set_title('Error Distribution by True Sentiment')
        
        # Text length distribution in errors
        if 'text_length' in errors_df.columns:
            axes[1,0].hist(errors_df['text_length'], bins=20, alpha=0.7, color='red', edgecolor='black')
            axes[1,0].set_title('Text Length Distribution in Errors')
            axes[1,0].set_xlabel('Text Length (words)')
            axes[1,0].set_ylabel('Frequency')
        
        # Error count by true label
        if 'true_label' in errors_df.columns:
            error_counts = errors_df['true_label'].value_counts().head(10)
            axes[1,1].bar(range(len(error_counts)), error_counts.values, color='coral')
            axes[1,1].set_title('Top 10 Error Counts by True Label')
            axes[1,1].set_xlabel('True Label')
            axes[1,1].set_ylabel('Error Count')
            # Truncate labels for readability
            labels = [label[:15] + '...' if len(label) > 15 else label for label in error_counts.index]
            axes[1,1].set_xticks(range(len(error_counts)))
            axes[1,1].set_xticklabels(labels, rotation=45, ha='right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'error_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
    
    def plot_label_distribution_analysis(self, label_encoder, y_true, y_pred):
        """Analyze and plot label distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Get all label names
        all_labels = label_encoder.classes_
        
        # True label distribution
        true_counts = np.bincount(y_true, minlength=len(all_labels))
        pred_counts = np.bincount(y_pred, minlength=len(all_labels))
        
        # Plot 1: True vs Predicted label distribution
        x = np.arange(len(all_labels))
        width = 0.35
        
        axes[0,0].bar(x - width/2, true_counts, width, label='True', alpha=0.8)
        axes[0,0].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        axes[0,0].set_title('True vs Predicted Label Distribution')
        axes[0,0].set_xlabel('Label Index')
        axes[0,0].set_ylabel('Count')
        axes[0,0].legend()
        
        # Plot 2: Zero-shot labels (labels with no training examples)
        zero_true = np.sum(true_counts == 0)
        zero_pred = np.sum(pred_counts == 0)
        
        axes[0,1].bar(['Labels with 0 True', 'Labels with 0 Pred'], [zero_true, zero_pred], 
                     color=['red', 'orange'])
        axes[0,1].set_title('Zero-Count Labels')
        axes[0,1].set_ylabel('Number of Labels')
        
        # Plot 3: Most and least frequent labels
        sorted_indices = np.argsort(true_counts)[::-1]
        top_10_indices = sorted_indices[:10]
        bottom_10_indices = sorted_indices[-10:]
        
        axes[1,0].bar(range(10), true_counts[top_10_indices], color='green', alpha=0.7)
        axes[1,0].set_title('Top 10 Most Frequent Labels (True)')
        axes[1,0].set_xlabel('Rank')
        axes[1,0].set_ylabel('Count')
        
        axes[1,1].bar(range(10), true_counts[bottom_10_indices], color='red', alpha=0.7)
        axes[1,1].set_title('Top 10 Least Frequent Labels (True)')
        axes[1,1].set_xlabel('Rank')
        axes[1,1].set_ylabel('Count')
        
        plt.tight_layout()
        plt.savefig(os.path.join(self.results_dir, 'label_distribution_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()
        
        # Print detailed statistics
        print(f"\n=== LABEL DISTRIBUTION ANALYSIS ===")
        print(f"Total labels: {len(all_labels)}")
        print(f"Labels with 0 true examples: {zero_true}")
        print(f"Labels with 0 predictions: {zero_pred}")
        print(f"Most frequent label: {all_labels[sorted_indices[0]]} ({true_counts[sorted_indices[0]]} examples)")
        print(f"Least frequent label: {all_labels[sorted_indices[-1]]} ({true_counts[sorted_indices[-1]]} examples)")
        
        return {
            'total_labels': len(all_labels),
            'zero_true_labels': zero_true,
            'zero_pred_labels': zero_pred,
            'true_distribution': true_counts,
            'pred_distribution': pred_counts
        }