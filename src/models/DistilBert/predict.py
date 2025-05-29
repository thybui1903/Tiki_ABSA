import torch
import time
import numpy as np

import os, certifi
os.environ["SSL_CERT_FILE"] = certifi.where()

from transformers import DistilBertTokenizer, DistilBertForSequenceClassification

from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.metrics import roc_auc_score, classification_report
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
from utils.helpers import get_device



class DistilBERTEvaluator:
    def __init__(self, model_name="distilbert-base-uncased"):
        self.device = get_device()
        
        # Load DistilBERT
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        
    def evaluate_performance(self, texts, labels, batch_size=16):
        """Đánh giá hiệu suất cơ bản"""
        predictions = []
        probabilities = []
        
        self.model.eval()
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i+batch_size]
                
                # Tokenize
                inputs = self.tokenizer(
                    batch_texts, 
                    truncation=True, 
                    padding=True, 
                    return_tensors="pt"
                ).to(self.device)
                
                # Predict
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                preds = torch.argmax(probs, dim=-1)
                
                predictions.extend(preds.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
        
        # Calculate metrics
        accuracy = accuracy_score(labels, predictions)
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average='weighted')
        
        # ROC-AUC for binary classification
        if len(np.unique(labels)) == 2:
            auc = roc_auc_score(labels, [p[1] for p in probabilities])
        else:
            auc = None
            
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'predictions': predictions,
            'probabilities': probabilities
        }
    
    def evaluate_efficiency(self, texts, batch_size=16, num_runs=5):
        """Đánh giá hiệu quả về tốc độ và bộ nhớ"""
        self.model.eval()
        
        # Measure inference time
        inference_times = []
        memory_usage = []
        
        for run in range(num_runs):
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
            start_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            start_time = time.time()
            
            with torch.no_grad():
                for i in range(0, len(texts), batch_size):
                    batch_texts = texts[i:i+batch_size]
                    inputs = self.tokenizer(
                        batch_texts, 
                        truncation=True, 
                        padding=True, 
                        return_tensors="pt"
                    ).to(self.device)
                    
                    outputs = self.model(**inputs)
            
            end_time = time.time()
            end_memory = psutil.Process().memory_info().rss / 1024 / 1024  # MB
            
            inference_times.append(end_time - start_time)
            memory_usage.append(end_memory - start_memory)
        
        avg_time = np.mean(inference_times)
        avg_memory = np.mean(memory_usage)
        throughput = len(texts) / avg_time  # samples per second
        
        return {
            'avg_inference_time': avg_time,
            'std_inference_time': np.std(inference_times),
            'avg_memory_usage': avg_memory,
            'throughput': throughput
        }
    
    
    def robustness_test(self, texts, labels, noise_level=0.1):
        """Test robustness với noisy inputs"""
        # Add character-level noise
        noisy_texts = []
        for text in texts:
            if np.random.random() < noise_level:
                # Randomly replace characters
                chars = list(text)
                if len(chars) > 0:
                    idx = np.random.randint(0, len(chars))
                    chars[idx] = np.random.choice(['a', 'e', 'i', 'o', 'u'])
                noisy_texts.append(''.join(chars))
            else:
                noisy_texts.append(text)
        
        # Evaluate on noisy data
        clean_results = self.evaluate_performance(texts, labels)
        noisy_results = self.evaluate_performance(noisy_texts, labels)
        
        robustness_score = noisy_results['accuracy'] / clean_results['accuracy']
        
        return {
            'clean_accuracy': clean_results['accuracy'],
            'noisy_accuracy': noisy_results['accuracy'],
            'robustness_score': robustness_score
        }
    
    def plot_confusion_matrix(self, labels, predictions, class_names=None):
        """Vẽ confusion matrix"""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('DistilBERT Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()
        
        return cm
    
    def comprehensive_report(self, texts, labels, class_names=None):
        """Báo cáo đánh giá toàn diện"""
        print("=== DistilBERT Comprehensive Evaluation Report ===\n")
        
        # Performance evaluation
        print("1. PERFORMANCE METRICS:")
        perf_results = self.evaluate_performance(texts, labels)
        print(f"   Accuracy: {perf_results['accuracy']:.4f}")
        print(f"   Precision: {perf_results['precision']:.4f}")
        print(f"   Recall: {perf_results['recall']:.4f}")
        print(f"   F1-Score: {perf_results['f1']:.4f}")
        if perf_results['auc']:
            print(f"   ROC-AUC: {perf_results['auc']:.4f}")
        
        # Efficiency evaluation
        print("\n2. EFFICIENCY METRICS:")
        eff_results = self.evaluate_efficiency(texts)
        print(f"   Avg Inference Time: {eff_results['avg_inference_time']:.4f}s")
        print(f"   Throughput: {eff_results['throughput']:.2f} samples/sec")
        print(f"   Memory Usage: {eff_results['avg_memory_usage']:.2f} MB")
        
        # Teacher comparison
        print("\n3. TEACHER COMPARISON:")
        comp_results = self.compare_with_teacher(texts, labels)
        print(f"   DistilBERT Accuracy: {comp_results['distilbert_accuracy']:.4f}")
        print(f"   BERT Accuracy: {comp_results['bert_accuracy']:.4f}")
        print(f"   Knowledge Retention: {comp_results['retention_ratio']:.2f}%")
        
        # Robustness test
        print("\n4. ROBUSTNESS TEST:")
        rob_results = self.robustness_test(texts, labels)
        print(f"   Clean Accuracy: {rob_results['clean_accuracy']:.4f}")
        print(f"   Noisy Accuracy: {rob_results['noisy_accuracy']:.4f}")
        print(f"   Robustness Score: {rob_results['robustness_score']:.4f}")
        
        # Confusion Matrix
        print("\n5. CONFUSION MATRIX:")
        self.plot_confusion_matrix(labels, perf_results['predictions'], class_names)
        
        return {
            'performance': perf_results,
            'efficiency': eff_results,
            'comparison': comp_results,
            'robustness': rob_results
        }

# Example usage
if __name__ == "__main__":
    # Sample data (replace with your actual data)
    sample_texts = [
        "This movie is fantastic!",
        "I hate this product.",
        "Average quality, nothing special.",
        "Excellent service and delivery.",
        "Terrible experience, would not recommend."
    ]
    sample_labels = [1, 0, 0, 1, 0]  # 1: positive, 0: negative
    class_names = ['Negative', 'Positive']
    
    # Initialize evaluator
    evaluator = DistilBERTEvaluator()
    
    # Run comprehensive evaluation
    results = evaluator.comprehensive_report(sample_texts, sample_labels, class_names)
    
    print("\n=== Evaluation Complete ===")