# utils.py - Modified for combined aspect-sentiment labels
import json
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import os
import re
from typing import List, Tuple, Dict, Any
from collections import Counter

class DataLoader:
    def __init__(self):
        self.label_encoder = LabelEncoder()
        
    def extract_labels_from_json(self, file_path: str) -> Tuple[List[str], List[str]]:
        """Extract texts and combined labels from JSON file"""
        texts = []
        labels = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            text = item['text']
            
            # Extract all labels from the item
            item_labels = []
            for label_info in item['labels']:
                if len(label_info) >= 4:
                    # label_info format: [start, end, span_text, combined_label]
                    combined_label = label_info[3]  # "Aspect#Sentiment"
                    item_labels.append(combined_label)
            
            # For each text, we can have multiple labels
            # For simplicity, let's take the first label or create a strategy
            if item_labels:
                # Option 1: Take the first label
                texts.append(text)
                labels.append(item_labels[0])
                
                # Option 2: If you want to handle multiple labels per text,
                # you might want to duplicate the text for each label
                # for label in item_labels:
                #     texts.append(text)
                #     labels.append(label)
        
        return texts, labels
    
    def prepare_combined_data(self, train_file: str, val_file: str, test_file: str) -> Dict[str, Tuple]:
        """Prepare data for combined aspect-sentiment classification"""
        
        # Load data from files
        train_texts, train_labels = self.extract_labels_from_json(train_file)
        val_texts, val_labels = self.extract_labels_from_json(val_file)
        test_texts, test_labels = self.extract_labels_from_json(test_file)
        
        # Combine all labels to fit the encoder
        all_labels = train_labels + val_labels + test_labels
        
        # Encode labels
        self.label_encoder.fit(all_labels)
        
        train_encoded = self.label_encoder.transform(train_labels)
        val_encoded = self.label_encoder.transform(val_labels)
        test_encoded = self.label_encoder.transform(test_labels)
        
        print(f"Found {len(self.label_encoder.classes_)} unique combined labels:")
        for i, label in enumerate(self.label_encoder.classes_[:10]):  # Show first 10
            print(f"  {i}: {label}")
        if len(self.label_encoder.classes_) > 10:
            print(f"  ... and {len(self.label_encoder.classes_) - 10} more")
        
        return {
            'train': (train_texts, train_encoded),
            'val': (val_texts, val_encoded), 
            'test': (test_texts, test_encoded)
        }
    
    def save_encoders(self, model_dir: str):
        """Save the label encoder"""
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def load_encoders(self, model_dir: str):
        """Load the label encoder"""
        with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
            self.label_encoder = pickle.load(f)

class TextVectorizer:
    def __init__(self, max_features=10000, ngram_range=(1, 2), min_df=2, max_df=0.95):
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=True,
            strip_accents='unicode',
            token_pattern=r'\b\w+\b'
        )
    
    def fit_transform(self, texts):
        """Fit vectorizer and transform texts"""
        return self.vectorizer.fit_transform(texts)
    
    def transform(self, texts):
        """Transform texts using fitted vectorizer"""
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        """Get feature names"""
        return self.vectorizer.get_feature_names_out()
    
    def save(self, model_dir: str):
        """Save the vectorizer"""
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'wb') as f:
            pickle.dump(self.vectorizer, f)
    
    def load(self, model_dir: str):
        """Load the vectorizer"""
        with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
            self.vectorizer = pickle.load(f)

# Additional utility functions for analysis
def analyze_label_distribution(labels: List[str]) -> Dict[str, Any]:
    """Analyze the distribution of combined labels"""
    label_counts = Counter(labels)
    
    # Extract aspects and sentiments
    aspects = []
    sentiments = []
    
    for label in labels:
        if '#' in label:
            aspect, sentiment = label.split('#', 1)
            aspects.append(aspect)
            sentiments.append(sentiment)
        else:
            aspects.append(label)
            sentiments.append('Unknown')
    
    aspect_counts = Counter(aspects)
    sentiment_counts = Counter(sentiments)
    
    return {
        'total_samples': len(labels),
        'unique_labels': len(label_counts),
        'label_distribution': dict(label_counts),
        'aspect_distribution': dict(aspect_counts),
        'sentiment_distribution': dict(sentiment_counts),
        'most_common_labels': label_counts.most_common(10),
        'most_common_aspects': aspect_counts.most_common(10),
        'most_common_sentiments': sentiment_counts.most_common(10)
    }

def preprocess_text(text: str) -> str:
    """Preprocess text for better classification"""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep Vietnamese characters
    # Keep letters, numbers, spaces, and basic punctuation
    text = re.sub(r'[^\w\s.,!?-]', ' ', text)
    
    # Remove multiple punctuation
    text = re.sub(r'[.,!?-]{2,}', '.', text)
    
    # Strip whitespace
    text = text.strip()
    
    return text

def split_combined_labels(labels: List[str]) -> Tuple[List[str], List[str]]:
    """Split combined labels into separate aspect and sentiment lists"""
    aspects = []
    sentiments = []
    
    for label in labels:
        if '#' in label:
            aspect, sentiment = label.split('#', 1)
            aspects.append(aspect)
            sentiments.append(sentiment)
        else:
            aspects.append(label)
            sentiments.append('Unknown')
    
    return aspects, sentiments

def create_error_analysis_df(y_true, y_pred, texts, label_encoder) -> pd.DataFrame:
    """Create DataFrame for error analysis"""
    errors_data = []
    
    for i, (true_idx, pred_idx, text) in enumerate(zip(y_true, y_pred, texts)):
        if true_idx != pred_idx:
            true_label = label_encoder.inverse_transform([true_idx])[0]
            pred_label = label_encoder.inverse_transform([pred_idx])[0]
            
            # Split combined labels
            if '#' in true_label:
                true_aspect, true_sentiment = true_label.split('#', 1)
            else:
                true_aspect, true_sentiment = true_label, 'Unknown'
                
            if '#' in pred_label:
                pred_aspect, pred_sentiment = pred_label.split('#', 1)
            else:
                pred_aspect, pred_sentiment = pred_label, 'Unknown'
            
            errors_data.append({
                'text': text,
                'text_length': len(text.split()),
                'true_label': true_label,
                'pred_label': pred_label,
                'true_aspect': true_aspect,
                'true_sentiment': true_sentiment,
                'pred_aspect': pred_aspect,
                'pred_sentiment': pred_sentiment,
                'aspect_correct': true_aspect == pred_aspect,
                'sentiment_correct': true_sentiment == pred_sentiment
            })
    
    return pd.DataFrame(errors_data)

def calculate_breakdown_metrics(y_true, y_pred, label_encoder) -> Dict[str, float]:
    """Calculate breakdown metrics for aspects and sentiments separately"""
    # Get label names
    true_labels = label_encoder.inverse_transform(y_true)
    pred_labels = label_encoder.inverse_transform(y_pred)
    
    # Split into aspects and sentiments
    true_aspects, true_sentiments = split_combined_labels(true_labels)
    pred_aspects, pred_sentiments = split_combined_labels(pred_labels)
    
    # Calculate accuracies
    aspect_accuracy = sum(1 for t, p in zip(true_aspects, pred_aspects) if t == p) / len(true_aspects)
    sentiment_accuracy = sum(1 for t, p in zip(true_sentiments, pred_sentiments) if t == p) / len(true_sentiments)
    
    return {
        'aspect_accuracy': aspect_accuracy,
        'sentiment_accuracy': sentiment_accuracy,
        'combined_accuracy': sum(1 for t, p in zip(y_true, y_pred) if t == p) / len(y_true)
    }

def print_data_summary(data_dict: Dict[str, Tuple]) -> None:
    """Print summary of the loaded data"""
    print("\n" + "="*50)
    print("DATA SUMMARY")
    print("="*50)
    
    for split_name, (texts, labels) in data_dict.items():
        print(f"\n{split_name.upper()} SET:")
        print(f"  Samples: {len(texts)}")
        print(f"  Unique labels: {len(set(labels))}")
        
        # Show label distribution
        label_counter = Counter(labels)
        print(f"  Most common labels:")
        for label_idx, count in label_counter.most_common(5):
            print(f"    {label_idx}: {count}")

def save_results_to_json(results: Dict[str, Any], filepath: str) -> None:
    """Save results dictionary to JSON file"""
    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {key: convert_numpy(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj
    
    converted_results = convert_numpy(results)
    
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(converted_results, f, indent=2, ensure_ascii=False)
    
    print(f"Results saved to: {filepath}")

def load_results_from_json(filepath: str) -> Dict[str, Any]:
    """Load results from JSON file"""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_class_weights(labels: List[int]) -> Dict[int, float]:
    """Calculate class weights for imbalanced datasets"""
    from sklearn.utils.class_weight import compute_class_weight
    
    unique_labels = np.unique(labels)
    class_weights = compute_class_weight(
        'balanced',
        classes=unique_labels,
        y=labels
    )
    
    return dict(zip(unique_labels, class_weights))

def validate_data_files(train_file: str, val_file: str, test_file: str) -> bool:
    """Validate that all required data files exist"""
    files = {'train': train_file, 'validation': val_file, 'test': test_file}
    
    missing_files = []
    for name, filepath in files.items():
        if not os.path.exists(filepath):
            missing_files.append(f"{name}: {filepath}")
    
    if missing_files:
        print("ERROR: Missing data files:")
        for missing in missing_files:
            print(f"  - {missing}")
        return False
    
    print("✓ All data files found")
    return True

# Example usage and testing
if __name__ == "__main__":
    # Test data loading
    print("Testing utils.py functionality...")
    
    # Test text preprocessing
    sample_texts = [
        "Sản phẩm rất tốt!!! Giao hàng nhanh.",
        "Chất lượng    kém, không như mong đợi...",
        "Giá cả hợp lý, sẽ mua lại!!!"
    ]
    
    print("\nText Preprocessing Test:")
    for text in sample_texts:
        processed = preprocess_text(text)
        print(f"Original: {text}")
        print(f"Processed: {processed}")
        print()
    
    # Test label analysis
    sample_labels = [
        "Chất lượng#Tích cực",
        "Giao hàng#Tiêu cực", 
        "Giá cả#Tích cực",
        "Chất lượng#Tích cực",
        "Dịch vụ#Trung tính"
    ]
    
    print("Label Distribution Analysis:")
    analysis = analyze_label_distribution(sample_labels)
    for key, value in analysis.items():
        print(f"{key}: {value}")
    
    print("\nutils.py testing completed successfully!")