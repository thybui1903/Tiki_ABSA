# evaluation/__init__.py
"""
DistilBERT Model Evaluation Module

Usage:
    from evaluation import DistilBERTModelEvaluator, evaluate_trained_model
    
    # Quick evaluation
    results = evaluate_trained_model(model_path, data_path)
    
    # Advanced evaluation
    evaluator = DistilBERTModelEvaluator(model, tokenizer)
    results = evaluator.evaluate_single_epoch(dataloader)
"""

from .evaluator import DistilBERTModelEvaluator


# Quick evaluation function
def evaluate_trained_model(model_path, data_path, **kwargs):
    """
    Quick evaluation function cho trained DistilBERT model
    
    Args:
        model_path: Path to trained model directory
        data_path: Path to evaluation data (.csv or .json)
        **kwargs: Additional configuration
        
    Returns:
        dict: Evaluation results
    """
    import torch
    import pandas as pd
    from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
    from torch.utils.data import DataLoader
    
    # Default config
    config = {
        'batch_size': 16,
        'max_length': 512,
        'num_labels': 2,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    config.update(kwargs)
    
    # Load model
    tokenizer = DistilBertTokenizer.from_pretrained(model_path)
    model = DistilBertForSequenceClassification.from_pretrained(model_path)
    model.to(config['device'])
    
    # Load data
    if data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
        texts = df['text'].tolist()
        labels = df['label'].tolist()
    else:
        raise ValueError("Only CSV format supported in quick evaluation")
    
    # Create dataset and dataloader
    from torch.utils.data import Dataset
    
    class SimpleDataset(Dataset):
        def __init__(self, texts, labels, tokenizer, max_length):
            self.texts = texts
            self.labels = labels
            self.tokenizer = tokenizer
            self.max_length = max_length
        
        def __len__(self):
            return len(self.texts)
        
        def __getitem__(self, idx):
            encoding = self.tokenizer(
                str(self.texts[idx]),
                truncation=True,
                padding='max_length',
                max_length=self.max_length,
                return_tensors='pt'
            )
            return {
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'labels': torch.tensor(self.labels[idx], dtype=torch.long)
            }
    
    dataset = SimpleDataset(texts, labels, tokenizer, config['max_length'])
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=False)
    
    # Evaluate
    evaluator = DistilBERTModelEvaluator(model, tokenizer, config['device'])
    results = evaluator.evaluate_single_epoch(dataloader)
    
    return results

__all__ = ['DistilBERTModelEvaluator', 'create_custom_metrics', 'evaluate_trained_model']