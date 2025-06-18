# detailed_evaluation.py
import torch
import numpy as np
import json
import os
from sklearn.metrics import classification_report, precision_recall_fscore_support, confusion_matrix
from transformers import AutoTokenizer
import pandas as pd
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns

# Import từ các module đã tạo
from config import ModelConfig, TrainingConfig
from utils import DataLoader, TextProcessor
from model import MultiTaskABSAModel, ABSADataset

class DetailedEvaluator:
    """Class để đánh giá chi tiết model đã train"""
    
    def __init__(self, model_path: str, config: ModelConfig):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model
        self.model = self._load_model(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        
        # Initialize processors
        self.data_loader = DataLoader(config)
        self.text_processor = TextProcessor()
        
    def _load_model(self, model_path: str) -> MultiTaskABSAModel:
        """Load trained model from checkpoint"""
        print(f"Loading model from {model_path}")

        # ✅ Không dùng weights_only vì file là dict
        checkpoint = torch.load(model_path, map_location=self.device)

        model = MultiTaskABSAModel(self.config)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()

        print(f"Best F1 from training: {checkpoint.get('best_f1', 'N/A')}")
        return model
        
    def evaluate_dataset(self, test_data_path: str) -> Dict:
        """Đánh giá model trên test dataset"""
        print(f"Evaluating on dataset: {test_data_path}")
        
        # Load test data
        test_data = self.data_loader.load_data(test_data_path)
        test_dataset = ABSADataset(test_data, self.data_loader)
        
        # Get predictions
        aspect_predictions, aspect_labels = self._get_predictions(test_dataset, task='aspect')
        sentiment_predictions, sentiment_labels = self._get_predictions(test_dataset, task='sentiment')
        
        # Calculate detailed metrics
        results = {
            'aspect_metrics': self._calculate_detailed_metrics(
                aspect_labels, aspect_predictions, 
                self.config.aspect_labels, 'Aspect'
            ),
            'sentiment_metrics': self._calculate_detailed_metrics(
                sentiment_labels, sentiment_predictions,
                self.config.sentiment_labels, 'Sentiment'
            )
        }
        
        return results
    
    def _get_predictions(self, dataset: ABSADataset, task: str) -> Tuple[np.ndarray, np.ndarray]:
        """Get predictions for specific task"""
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for i in range(len(dataset)):
                sample = dataset[i]
                
                # Prepare input
                input_ids = sample['input_ids'].unsqueeze(0).to(self.device)
                attention_mask = sample['attention_mask'].unsqueeze(0).to(self.device)
                
                # Get model output
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                
                if task == 'aspect':
                    predictions = torch.sigmoid(outputs['aspect_logits']) > 0.5
                    labels = sample['aspect_labels']
                else:  # sentiment
                    predictions = torch.sigmoid(outputs['sentiment_logits']) > 0.5
                    labels = sample['sentiment_labels']
                
                all_predictions.append(predictions.cpu().numpy()[0])
                all_labels.append(labels.numpy())
        
        return np.array(all_predictions), np.array(all_labels)
    
    def _calculate_detailed_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                                  labels: List[str], task_name: str) -> Dict:
        """Tính toán metrics chi tiết cho từng class"""
        print(f"\n=== {task_name} Classification Report ===")
        
        # Classification report
        report = classification_report(
            y_true, y_pred, 
            target_names=labels,
            output_dict=True,
            zero_division=0
        )
        
        # Print detailed report
        print(classification_report(
            y_true, y_pred,
            target_names=labels,
            zero_division=0
        ))
        
        # Calculate per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true, y_pred, average=None, zero_division=0
        )
        
        # Organize results
        detailed_metrics = {
            'overall': {
                'micro_avg': report['micro avg'],
                'macro_avg': report['macro avg'],
                'weighted_avg': report['weighted avg']
            },
            'per_class': {}
        }
        
        # Per-class metrics
        for i, label in enumerate(labels):
            detailed_metrics['per_class'][label] = {
                'precision': float(precision[i]),
                'recall': float(recall[i]),
                'f1_score': float(f1[i]),
                'support': int(support[i])
            }
        
        return detailed_metrics
    
    def create_confusion_matrices(self, test_data_path: str, save_dir: str = "./evaluation_results"):
        """Tạo confusion matrix cho từng task"""
        os.makedirs(save_dir, exist_ok=True)
        
        test_data = self.data_loader.load_data(test_data_path)
        test_dataset = ABSADataset(test_data, self.data_loader)
        
        # Get predictions
        aspect_predictions, aspect_labels = self._get_predictions(test_dataset, task='aspect')
        sentiment_predictions, sentiment_labels = self._get_predictions(test_dataset, task='sentiment')
        
        # Create confusion matrices for each class
        self._plot_confusion_matrices(
            aspect_labels, aspect_predictions, 
            self.config.aspect_labels, 'Aspect', save_dir
        )
        
        self._plot_confusion_matrices(
            sentiment_labels, sentiment_predictions,
            self.config.sentiment_labels, 'Sentiment', save_dir
        )

        """Tạo confusion matrix tổng hợp cho tất cả nhãn aspect + sentiment"""
        # Combine predictions and labels
        y_pred_combined = np.concatenate([aspect_predictions, sentiment_predictions], axis=1)
        y_true_combined = np.concatenate([aspect_labels, sentiment_labels], axis=1)

        # Combine label names
        all_labels = self.config.aspect_labels + self.config.sentiment_labels
        # Plot combined confusion matrix
        self._plot_combined_confusion_matrix(y_true_combined, y_pred_combined, all_labels, save_dir)

    def _plot_combined_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray,
                                     labels: List[str], save_dir: str):
        """
        Plot confusion matrix cho toàn bộ nhãn aspect + sentiment (multi-label setting, binary per label)
        """
        n_classes = len(labels)
        cm_all = []

        for i in range(n_classes):
            cm = confusion_matrix(y_true[:, i], y_pred[:, i], labels=[0, 1])
            tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, cm[0][0] if cm.size == 1 else 0)
            cm_all.append([tp, fn, fp, tn])  # true positive, false negative, false positive, true negative

        df_cm = pd.DataFrame(cm_all, columns=['TP', 'FN', 'FP', 'TN'], index=labels)

        plt.figure(figsize=(10, 0.5 * len(labels) + 2))
        sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix for All Labels')
        plt.xlabel('Metric')
        plt.ylabel('Labels')
        plt.tight_layout()

        output_path = os.path.join(save_dir, 'combined_confusion_matrix.png')
        plt.savefig(output_path, dpi=300)
        plt.close()

        print(f"Combined confusion matrix saved to {output_path}")
        
        
    def _plot_confusion_matrices(self, y_true: np.ndarray, y_pred: np.ndarray,
                                labels: List[str], task_name: str, save_dir: str):
        """Plot confusion matrix for each class in multi-label setting"""
        n_classes = len(labels)
        fig, axes = plt.subplots(1, n_classes, figsize=(5*n_classes, 4))
        
        if n_classes == 1:
            axes = [axes]
        
        for i, label in enumerate(labels):
            # Create binary confusion matrix for each class
            cm = confusion_matrix(y_true[:, i], y_pred[:, i])
            
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['Negative', 'Positive'],
                       yticklabels=['Negative', 'Positive'],
                       ax=axes[i])
            axes[i].set_title(f'{task_name}: {label}')
            axes[i].set_xlabel('Predicted')
            axes[i].set_ylabel('Actual')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{task_name.lower()}_confusion_matrices.png'),
                   dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"{task_name} confusion matrices saved to {save_dir}")
    
    def analyze_errors(self, test_data_path: str, save_path: str = "./evaluation_results/error_analysis.json"):
        """Phân tích lỗi chi tiết"""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        test_data = self.data_loader.load_data(test_data_path)
        test_dataset = ABSADataset(test_data, self.data_loader)
        
        error_analysis = {
            'aspect_errors': [],
            'sentiment_errors': [],
            'statistics': {}
        }
        
        aspect_predictions, aspect_labels = self._get_predictions(test_dataset, task='aspect')
        sentiment_predictions, sentiment_labels = self._get_predictions(test_dataset, task='sentiment')
        
        # Analyze errors
        for i in range(len(test_data)):
            sample = test_data[i]
            text = sample['text']
            
            # Aspect errors
            aspect_true = aspect_labels[i]
            aspect_pred = aspect_predictions[i]
            
            if not np.array_equal(aspect_true, aspect_pred):
                error_analysis['aspect_errors'].append({
                    'text': text,
                    'true_labels': [self.config.aspect_labels[j] for j in range(len(aspect_true)) if aspect_true[j] == 1],
                    'predicted_labels': [self.config.aspect_labels[j] for j in range(len(aspect_pred)) if aspect_pred[j] == 1],
                    'original_labels': sample['labels']
                })
            
            # Sentiment errors
            sentiment_true = sentiment_labels[i]
            sentiment_pred = sentiment_predictions[i]
            
            if not np.array_equal(sentiment_true, sentiment_pred):
                error_analysis['sentiment_errors'].append({
                    'text': text,
                    'true_labels': [self.config.sentiment_labels[j] for j in range(len(sentiment_true)) if sentiment_true[j] == 1],
                    'predicted_labels': [self.config.sentiment_labels[j] for j in range(len(sentiment_pred)) if sentiment_pred[j] == 1],
                    'original_labels': sample['labels']
                })
        
        # Error statistics
        error_analysis['statistics'] = {
            'total_samples': len(test_data),
            'aspect_errors': len(error_analysis['aspect_errors']),
            'sentiment_errors': len(error_analysis['sentiment_errors']),
            'aspect_error_rate': len(error_analysis['aspect_errors']) / len(test_data),
            'sentiment_error_rate': len(error_analysis['sentiment_errors']) / len(test_data)
        }
        
        # Save error analysis
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(error_analysis, f, ensure_ascii=False, indent=2)
        
        print(f"Error analysis saved to {save_path}")
        return error_analysis
    
    def generate_evaluation_report(self, test_data_path: str, output_dir: str = "./evaluation_results"):
        """Tạo báo cáo đánh giá hoàn chỉnh"""
        os.makedirs(output_dir, exist_ok=True)
        
        print("Generating comprehensive evaluation report...")
        
        # 1. Detailed metrics
        results = self.evaluate_dataset(test_data_path)
        
        # 2. Confusion matrices
        self.create_confusion_matrices(test_data_path, output_dir)
        
        # 3. Error analysis
        error_analysis = self.analyze_errors(test_data_path, 
                                           os.path.join(output_dir, "error_analysis.json"))
        
        # 4. Save detailed metrics
        with open(os.path.join(output_dir, "detailed_metrics.json"), 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        
        # 5. Create summary report
        self._create_summary_report(results, error_analysis, output_dir)
        
        print(f"Complete evaluation report saved to {output_dir}")
        return results
    
    """def create_confusion_matrices(self, test_data_path: str, save_dir: str = "./evaluation_results"):
        # Tạo confusion matrix gộp giữa aspect và sentiment
        os.makedirs(save_dir, exist_ok=True)

        test_data = self.data_loader.load_data(test_data_path)
        test_dataset = ABSADataset(test_data, self.data_loader)

        # Get predictions
        aspect_preds, _ = self._get_predictions(test_dataset, task='aspect')
        sentiment_preds, _ = self._get_predictions(test_dataset, task='sentiment')

        aspect_labels = self.config.aspect_labels
        sentiment_labels = self.config.sentiment_labels

        combined_labels = []
        y_true_combined = []
        y_pred_combined = []

        for i in range(len(test_dataset)):
            for a_idx, a_label in enumerate(aspect_labels):
                if test_dataset[i]['aspect_labels'][a_idx] == 1:
                    for s_idx, s_label in enumerate(sentiment_labels):
                        if test_dataset[i]['sentiment_labels'][s_idx] == 1:
                            true_label = f"{a_label}#{s_label}"
                            y_true_combined.append(true_label)

            for a_idx, a_label in enumerate(aspect_labels):
                if aspect_preds[i][a_idx] == 1:
                    for s_idx, s_label in enumerate(sentiment_labels):
                        if sentiment_preds[i][s_idx] == 1:
                            pred_label = f"{a_label}#{s_label}"
                            y_pred_combined.append(pred_label)

        # Lấy danh sách các nhãn xuất hiện thật sự để sắp xếp theo thứ tự rõ ràng
        all_labels = sorted(set(y_true_combined + y_pred_combined))

        # Tạo confusion matrix
        cm = confusion_matrix(y_true_combined, y_pred_combined, labels=all_labels)

        # Plot
        plt.figure(figsize=(max(10, len(all_labels) * 0.5), max(8, len(all_labels) * 0.5)))
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=all_labels, yticklabels=all_labels, cmap='Blues')
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.title("Combined Aspect-Sentiment Confusion Matrix")
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        plt.tight_layout()

        save_path = os.path.join(save_dir, "combined_confusion_matrix.png")
        plt.savefig(save_path, dpi=300)
        plt.close()

        print(f"Combined confusion matrix saved to {save_path}")
"""
    
    def _create_summary_report(self, results: Dict, error_analysis: Dict, output_dir: str):
        """Tạo báo cáo tóm tắt"""
        report_lines = []
        report_lines.append("=== MODEL EVALUATION SUMMARY ===\n")
        
        # Aspect metrics summary
        report_lines.append("ASPECT CLASSIFICATION:")
        aspect_metrics = results['aspect_metrics']
        report_lines.append(f"  Overall F1 (Macro): {aspect_metrics['overall']['macro_avg']['f1-score']:.4f}")
        report_lines.append(f"  Overall F1 (Weighted): {aspect_metrics['overall']['weighted_avg']['f1-score']:.4f}")
        report_lines.append(f"  Overall Precision (Macro): {aspect_metrics['overall']['macro_avg']['precision']:.4f}")
        report_lines.append(f"  Overall Recall (Macro): {aspect_metrics['overall']['macro_avg']['recall']:.4f}")
        
        report_lines.append("\n  Per-Class Metrics:")
        for aspect_name, metrics in aspect_metrics['per_class'].items():
            report_lines.append(f"    {aspect_name}: F1={metrics['f1_score']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
        
        # Sentiment metrics summary
        report_lines.append("\nSENTIMENT CLASSIFICATION:")
        sentiment_metrics = results['sentiment_metrics']
        report_lines.append(f"  Overall F1 (Macro): {sentiment_metrics['overall']['macro_avg']['f1-score']:.4f}")
        report_lines.append(f"  Overall F1 (Weighted): {sentiment_metrics['overall']['weighted_avg']['f1-score']:.4f}")
        report_lines.append(f"  Overall Precision (Macro): {sentiment_metrics['overall']['macro_avg']['precision']:.4f}")
        report_lines.append(f"  Overall Recall (Macro): {sentiment_metrics['overall']['macro_avg']['recall']:.4f}")
        
        report_lines.append("\n  Per-Class Metrics:")
        for sentiment_name, metrics in sentiment_metrics['per_class'].items():
            report_lines.append(f"    {sentiment_name}: F1={metrics['f1_score']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}")
        
        # Error statistics
        report_lines.append("\nERROR ANALYSIS:")
        stats = error_analysis['statistics']
        report_lines.append(f"  Total samples: {stats['total_samples']}")
        report_lines.append(f"  Aspect errors: {stats['aspect_errors']} ({stats['aspect_error_rate']:.2%})")
        report_lines.append(f"  Sentiment errors: {stats['sentiment_errors']} ({stats['sentiment_error_rate']:.2%})")
        
        # Save report
        report_content = "\n".join(report_lines)
        with open(os.path.join(output_dir, "summary_report.txt"), 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print("\n" + report_content)

def main():
    """Main function để chạy evaluation"""
    
    # Khởi tạo config
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Đường dẫn model đã train
    model_path = os.path.join(training_config.model_save_path, "C:/Users/DELL/Tiki_ABSA/src/models/DistilBert_MultiTasking/checkpoints/best_model.pt")
    
    # Đường dẫn test data
    test_data_path = training_config.test_data_path
    
    # Kiểm tra file tồn tại
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    
    if not os.path.exists(test_data_path):
        print(f"Error: Test data file not found at {test_data_path}")
        return
    
    # Khởi tạo evaluator
    evaluator = DetailedEvaluator(model_path, model_config)
    
    # Chạy evaluation hoàn chỉnh
    results = evaluator.generate_evaluation_report(test_data_path)
    
    print("\n=== EVALUATION COMPLETED ===")
    print("Check './evaluation_results/' folder for detailed results:")
    print("  - detailed_metrics.json: Chi tiết metrics từng class")
    print("  - error_analysis.json: Phân tích lỗi")
    print("  - summary_report.txt: Báo cáo tóm tắt")
    print("  - confusion_matrices.png: Confusion matrices")

if __name__ == "__main__":
    main()

# Ví dụ sử dụng riêng lẻ:
"""
# 1. Chỉ tính metrics chi tiết
evaluator = DetailedEvaluator("./saved_models/best_model.pt", ModelConfig())
results = evaluator.evaluate_dataset("./data/test.json")

# 2. Chỉ tạo confusion matrix
evaluator.create_confusion_matrices("./data/test.json")

# 3. Chỉ phân tích lỗi
error_analysis = evaluator.analyze_errors("./data/test.json")

# 4. Báo cáo hoàn chỉnh
full_results = evaluator.generate_evaluation_report("./data/test.json")
"""