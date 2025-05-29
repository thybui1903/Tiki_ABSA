import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, average_precision_score, cohen_kappa_score,
    classification_report, confusion_matrix
)
from sklearn.model_selection import cross_val_score
import seaborn as sns
from collections import defaultdict

class DistilBERTModelEvaluator:
    def __init__(self, model, tokenizer, device='cuda'):
        self.model = model
        self.tokenizer = tokenizer 
        self.device = device
        self.evaluation_history = defaultdict(list)
        
    def evaluate_single_epoch(self, dataloader, epoch=None):
        """Đánh giá model cho một epoch"""
        self.model.eval()
        total_loss = 0
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                total_loss += outputs.loss.item()
                
                # Get predictions
                logits = outputs.logits
                probabilities = torch.softmax(logits, dim=-1)
                predictions = torch.argmax(logits, dim=-1)
                
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        avg_loss = total_loss / len(dataloader)
        
        # Calculate metrics
        metrics = self.calculate_comprehensive_metrics(
            all_labels, all_predictions, all_probabilities
        )
        metrics['loss'] = avg_loss
        
        # Store in history
        if epoch is not None:
            for key, value in metrics.items():
                self.evaluation_history[key].append(value)
        
        return metrics
    
    def calculate_comprehensive_metrics(self, y_true, y_pred, y_prob=None):
        """Tính toán tất cả các metrics quan trọng"""
        metrics = {}
        
        # Basic Classification Metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro')
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted')
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro')
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted')
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro')
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted')
        
        # Cohen's Kappa - đo độ đồng thuận
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        
        # ROC-AUC và Average Precision (nếu có probabilities)
        if y_prob is not None:
            n_classes = len(np.unique(y_true))
            
            if n_classes == 2:
                # Binary classification
                metrics['roc_auc'] = roc_auc_score(y_true, [p[1] for p in y_prob])
                metrics['avg_precision'] = average_precision_score(y_true, [p[1] for p in y_prob])
            else:
                # Multi-class classification
                try:
                    metrics['roc_auc_ovr'] = roc_auc_score(y_true, y_prob, multi_class='ovr')
                    metrics['roc_auc_ovo'] = roc_auc_score(y_true, y_prob, multi_class='ovo')
                except:
                    pass
        
        # Per-class metrics
        per_class_f1 = f1_score(y_true, y_pred, average=None)
        per_class_precision = precision_score(y_true, y_pred, average=None)
        per_class_recall = recall_score(y_true, y_pred, average=None)
        
        metrics['per_class_f1'] = per_class_f1
        metrics['per_class_precision'] = per_class_precision  
        metrics['per_class_recall'] = per_class_recall
        
        # Class balance metrics
        metrics['min_class_f1'] = np.min(per_class_f1)
        metrics['max_class_f1'] = np.max(per_class_f1)
        metrics['std_class_f1'] = np.std(per_class_f1)
        
        return metrics
    
    def early_stopping_check(self, patience=3, min_delta=0.001, metric='f1_weighted'):
        """Kiểm tra early stopping"""
        if len(self.evaluation_history[metric]) < patience + 1:
            return False
        
        recent_scores = self.evaluation_history[metric][-patience-1:]
        best_score = max(recent_scores[:-1])
        current_score = recent_scores[-1]
        
        # Nếu không cải thiện đủ trong patience epochs
        if current_score - best_score < min_delta:
            return True
        return False
    
    def find_best_epoch(self, metric='f1_weighted'):
        """Tìm epoch tốt nhất dựa trên metric"""
        if metric not in self.evaluation_history:
            return None
        
        scores = self.evaluation_history[metric]
        best_epoch = np.argmax(scores)
        best_score = scores[best_epoch]
        
        return {
            'best_epoch': best_epoch,
            'best_score': best_score,
            'metric': metric
        }
    
    def plot_training_curves(self, metrics=['loss', 'accuracy', 'f1_weighted']):
        """Vẽ đường cong training"""
        fig, axes = plt.subplots(1, len(metrics), figsize=(5*len(metrics), 4))
        
        if len(metrics) == 1:
            axes = [axes]
        
        for i, metric in enumerate(metrics):
            if metric in self.evaluation_history:
                epochs = range(len(self.evaluation_history[metric]))
                axes[i].plot(epochs, self.evaluation_history[metric], 'b-', linewidth=2)
                axes[i].set_title(f'{metric.replace("_", " ").title()}')
                axes[i].set_xlabel('Epoch')
                axes[i].set_ylabel(metric)
                axes[i].grid(True, alpha=0.3)
                
                # Mark best epoch
                if metric != 'loss':
                    best_idx = np.argmax(self.evaluation_history[metric])
                else:
                    best_idx = np.argmin(self.evaluation_history[metric])
                
                axes[i].axvline(x=best_idx, color='red', linestyle='--', alpha=0.7)
                axes[i].scatter([best_idx], [self.evaluation_history[metric][best_idx]], 
                              color='red', s=100, zorder=5)
        
        plt.tight_layout()
        plt.show()
    
    def compare_models(self, model_results, model_names=None):
        """So sánh nhiều models"""
        if model_names is None:
            model_names = [f'Model_{i+1}' for i in range(len(model_results))]
        
        # Key metrics để so sánh
        key_metrics = ['accuracy', 'f1_weighted', 'precision_weighted', 'recall_weighted', 'cohen_kappa']
        
        comparison_df = []
        for i, results in enumerate(model_results):
            row = {'Model': model_names[i]}
            for metric in key_metrics:
                if metric in results:
                    row[metric] = results[metric]
            comparison_df.append(row)
        
        return comparison_df
    
    def detailed_classification_report(self, y_true, y_pred, class_names=None):
        """Báo cáo phân loại chi tiết"""
        print("=== DETAILED CLASSIFICATION REPORT ===\n")
        
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro')
        f1_weighted = f1_score(y_true, y_pred, average='weighted')
        
        print(f"Overall Accuracy: {accuracy:.4f}")
        print(f"Macro F1-Score: {f1_macro:.4f}")
        print(f"Weighted F1-Score: {f1_weighted:.4f}")
        print(f"Cohen's Kappa: {cohen_kappa_score(y_true, y_pred):.4f}")
        
        # Per-class report
        print("\nPer-Class Metrics:")
        print(classification_report(y_true, y_pred, target_names=class_names))
        
        # Confusion Matrix
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'confusion_matrix': cm
        }
    
    def model_selection_summary(self):
        """Tóm tắt để chọn model tốt nhất"""
        print("=== MODEL SELECTION SUMMARY ===\n")
        
        # Find best performing epochs for different metrics
        key_metrics = ['f1_weighted', 'accuracy', 'f1_macro', 'cohen_kappa']
        
        for metric in key_metrics:
            if metric in self.evaluation_history:
                best_info = self.find_best_epoch(metric)
                print(f"Best {metric}: {best_info['best_score']:.4f} at epoch {best_info['best_epoch']}")
        
        # Recommendations
        print("\nRECOMMENDATIONS:")
        
        # Check for overfitting
        if len(self.evaluation_history.get('loss', [])) > 5:
            recent_losses = self.evaluation_history['loss'][-5:]
            if all(x <= y for x, y in zip(recent_losses, recent_losses[1:])):
                print("⚠️  Potential overfitting detected - loss still decreasing")
            else:
                print("✅ Loss has stabilized")
        
        # Check metric stability
        if 'f1_weighted' in self.evaluation_history and len(self.evaluation_history['f1_weighted']) > 3:
            recent_f1 = self.evaluation_history['f1_weighted'][-3:]
            if np.std(recent_f1) < 0.01:
                print("✅ F1 score has converged")
            else:
                print("⚠️  F1 score still fluctuating")
        
        # Best overall recommendation
        if 'f1_weighted' in self.evaluation_history:
            best_f1_epoch = self.find_best_epoch('f1_weighted')
            print(f"\n🏆 RECOMMENDED MODEL: Epoch {best_f1_epoch['best_epoch']} "
                  f"(F1-Weighted: {best_f1_epoch['best_score']:.4f})")

# Example usage for model selection
def select_best_distilbert_model(models_and_results):
    """
    models_and_results: list of tuples (model, evaluation_results)
    """
    
    print("=== DISTILBERT MODEL SELECTION ===\n")
    
    best_model = None
    best_score = 0
    best_metrics = None
    
    comparison_data = []
    
    for i, (model, results) in enumerate(models_and_results):
        model_name = f"Model_{i+1}"
        
        # Primary metric cho selection (có thể thay đổi)
        primary_score = results.get('f1_weighted', 0)
        
        comparison_data.append({
            'Model': model_name,
            'F1_Weighted': primary_score,
            'Accuracy': results.get('accuracy', 0),
            'F1_Macro': results.get('f1_macro', 0),
            'Cohen_Kappa': results.get('cohen_kappa', 0),
            'Loss': results.get('loss', float('inf'))
        })
        
        if primary_score > best_score:
            best_score = primary_score
            best_model = model
            best_metrics = results
    
    # Display comparison table
    import pandas as pd
    df = pd.DataFrame(comparison_data)
    print("Model Comparison:")
    print(df.round(4))
    
    print(f"\n🏆 BEST MODEL: {df.loc[df['F1_Weighted'].idxmax(), 'Model']}")
    print(f"Best F1-Weighted Score: {best_score:.4f}")
    
    return best_model, best_metrics

# Advanced model selection with multiple criteria
def advanced_model_selection(evaluation_results, weights=None):
    """
    Chọn model dựa trên weighted combination của nhiều metrics
    """
    if weights is None:
        weights = {
            'f1_weighted': 0.4,
            'accuracy': 0.3, 
            'cohen_kappa': 0.2,
            'loss': -0.1  # negative weight vì loss càng thấp càng tốt
        }
    
    scores = []
    for results in evaluation_results:
        weighted_score = 0
        for metric, weight in weights.items():
            if metric in results:
                if metric == 'loss':
                    # Normalize loss (assume lower is better)
                    normalized_value = 1 / (1 + results[metric])
                else:
                    normalized_value = results[metric]
                weighted_score += weight * normalized_value
        scores.append(weighted_score)
    
    best_idx = np.argmax(scores)
    
    print(f"Advanced Selection - Best Model Index: {best_idx}")
    print(f"Weighted Score: {scores[best_idx]:.4f}")
    
    return best_idx, scores