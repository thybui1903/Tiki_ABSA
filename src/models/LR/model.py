# model.py - Fixed log loss calculation for combined aspect-sentiment labels
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, classification_report, log_loss
import numpy as np
import pickle
import os
from typing import Dict, Tuple, Any

class ABSALogisticRegression:
    def __init__(self, C=1.0, max_iter=1000, random_state=42):
        self.C = C
        self.max_iter = max_iter
        self.random_state = random_state
        
        # Single model for combined aspect-sentiment classification
        self.model = LogisticRegression(
            C=C, max_iter=max_iter, random_state=random_state
        )
        
        self.n_classes = None
        self.classes_ = None
        
    def fit(self, X, y):
        """Train the combined aspect-sentiment model"""
        self.model.fit(X, y)
        self.n_classes = len(np.unique(y))
        self.classes_ = self.model.classes_
        return self
    
    def predict(self, X):
        """Predict combined aspect-sentiment labels"""
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def calculate_loss(self, y_true, y_proba):
        """Calculate log loss with proper handling of missing classes"""
        try:
            # Get all possible classes from the model
            all_classes = self.model.classes_
            
            # Create a mapping from class to index
            class_to_idx = {cls: idx for idx, cls in enumerate(all_classes)}
            
            # Convert y_true to proper format
            y_true_encoded = np.array([class_to_idx.get(label, 0) for label in y_true])
            
            # Ensure y_proba has the right shape
            if y_proba.shape[1] != len(all_classes):
                print(f"Warning: Probability matrix shape mismatch. Expected {len(all_classes)} classes, got {y_proba.shape[1]}")
                return 0.0
            
            # Calculate log loss with explicit labels
            return log_loss(y_true_encoded, y_proba, labels=list(range(len(all_classes))))
            
        except Exception as e:
            print(f"Warning: Could not calculate log loss: {e}")
            return 0.0
    
    def evaluate(self, X, y_true):
        """Comprehensive evaluation with all metrics"""
        # Get predictions
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)
        
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_micro = f1_score(y_true, y_pred, average='micro', zero_division=0)
        
        # Calculate loss with proper handling
        loss = self.calculate_loss(y_true, y_proba)
        
        # Classification report with proper labels
        try:
            report = classification_report(
                y_true, y_pred, 
                output_dict=True, 
                zero_division=0,
                labels=self.model.classes_  # Explicitly specify all possible labels
            )
        except Exception as e:
            print(f"Warning: Could not generate classification report: {e}")
            report = {}
        
        return {
            'accuracy': accuracy,
            'f1_macro': f1_macro,
            'f1_weighted': f1_weighted,
            'f1_micro': f1_micro,
            'loss': loss,
            'predictions': y_pred,
            'probabilities': y_proba,
            'classification_report': report,
            
            # Additional metrics for analysis
            'support': np.bincount(y_true, minlength=len(self.model.classes_)),
            'classes': self.classes_
        }
    
    def get_aspect_sentiment_breakdown(self, y_true, y_pred):
        """Break down results by aspect and sentiment separately"""
        # Convert numeric labels back to string labels if needed
        if hasattr(self, 'label_encoder'):
            true_labels = self.label_encoder.inverse_transform(y_true)
            pred_labels = self.label_encoder.inverse_transform(y_pred)
        else:
            # Assume they're already in string format or handle accordingly
            true_labels = y_true
            pred_labels = y_pred
        
        # Extract aspects and sentiments from combined labels
        aspects_true = []
        sentiments_true = []
        aspects_pred = []
        sentiments_pred = []
        
        for true_label, pred_label in zip(true_labels, pred_labels):
            # Split combined labels (assuming format: "Aspect#Sentiment")
            true_parts = str(true_label).split('#')
            pred_parts = str(pred_label).split('#')
            
            if len(true_parts) == 2 and len(pred_parts) == 2:
                aspects_true.append(true_parts[0])
                sentiments_true.append(true_parts[1])
                aspects_pred.append(pred_parts[0])
                sentiments_pred.append(pred_parts[1])
        
        if aspects_true:  # Only calculate if we have valid data
            aspect_accuracy = accuracy_score(aspects_true, aspects_pred)
            sentiment_accuracy = accuracy_score(sentiments_true, sentiments_pred)
            
            aspect_f1_macro = f1_score(aspects_true, aspects_pred, average='macro', zero_division=0)
            sentiment_f1_macro = f1_score(sentiments_true, sentiments_pred, average='macro', zero_division=0)
            
            return {
                'aspect_accuracy': aspect_accuracy,
                'sentiment_accuracy': sentiment_accuracy,
                'aspect_f1_macro': aspect_f1_macro,
                'sentiment_f1_macro': sentiment_f1_macro,
                'aspects_true': aspects_true,
                'sentiments_true': sentiments_true,
                'aspects_pred': aspects_pred,
                'sentiments_pred': sentiments_pred
            }
        
        return None
    
    def save(self, model_dir: str):
        """Save the trained model"""
        os.makedirs(model_dir, exist_ok=True)
        
        with open(os.path.join(model_dir, 'model.pkl'), 'wb') as f:
            pickle.dump(self.model, f)
            
        # Save model info
        model_info = {
            'C': self.C,
            'max_iter': self.max_iter,
            'random_state': self.random_state,
            'n_classes': self.n_classes,
            'classes_': self.classes_.tolist() if self.classes_ is not None else None
        }
        
        with open(os.path.join(model_dir, 'model_info.pkl'), 'wb') as f:
            pickle.dump(model_info, f)
    
    def load(self, model_dir: str):
        """Load the trained model"""
        with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
            self.model = pickle.load(f)
            
        with open(os.path.join(model_dir, 'model_info.pkl'), 'rb') as f:
            model_info = pickle.load(f)
            self.C = model_info['C']
            self.max_iter = model_info['max_iter']
            self.random_state = model_info['random_state']
            self.n_classes = model_info['n_classes']
            self.classes_ = np.array(model_info['classes_']) if model_info['classes_'] else None
        
        return self