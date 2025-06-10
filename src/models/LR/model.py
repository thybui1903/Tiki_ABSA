from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import os
from sklearn.metrics import accuracy_score, classification_report, f1_score

class ABSALogisticRegression:

    def __init__(self, C=1.0, max_iter=1000, random_state=42):
        base_model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            random_state=random_state
        )

        # Dùng OneVsRest cho đa lớp, tránh FutureWarning
        self.aspect_model = OneVsRestClassifier(base_model)
        self.sentiment_model = OneVsRestClassifier(base_model)

        self.is_fitted = False
    
    def fit(self, X_train, y_aspect_train, y_sentiment_train):
        """Training models"""
        print("Training aspect detection model...")
        self.aspect_model.fit(X_train, y_aspect_train)
        
        print("Training sentiment classification model...")
        self.sentiment_model.fit(X_train, y_sentiment_train)
        
        self.is_fitted = True
        return self
    
    def predict(self, X):
        """Predict aspects and sentiments"""
        aspect_pred = self.aspect_model.predict(X)
        sentiment_pred = self.sentiment_model.predict(X)
        return aspect_pred, sentiment_pred
    
    def predict_proba(self, X):
        """Predict probabilities"""
        aspect_proba = self.aspect_model.predict_proba(X)
        sentiment_proba = self.sentiment_model.predict_proba(X)
        return aspect_proba, sentiment_proba
    
    def evaluate(self, X, y_aspect, y_sentiment):
        """Đánh giá model"""
        aspect_pred, sentiment_pred = self.predict(X)

        aspect_accuracy = accuracy_score(y_aspect, aspect_pred)
        sentiment_accuracy = accuracy_score(y_sentiment, sentiment_pred)

        # F1-score
        aspect_f1 = {
            'macro': f1_score(y_aspect, aspect_pred, average='macro', zero_division=0),
            'weighted': f1_score(y_aspect, aspect_pred, average='weighted', zero_division=0),
            'micro': f1_score(y_aspect, aspect_pred, average='micro', zero_division=0),
        }

        sentiment_f1 = {
            'macro': f1_score(y_sentiment, sentiment_pred, average='macro', zero_division=0),
            'weighted': f1_score(y_sentiment, sentiment_pred, average='weighted', zero_division=0),
            'micro': f1_score(y_sentiment, sentiment_pred, average='micro', zero_division=0),
        }

        aspect_report = classification_report(
            y_aspect, aspect_pred, output_dict=True, zero_division=0
        )
        sentiment_report = classification_report(
            y_sentiment, sentiment_pred, output_dict=True, zero_division=0
        )

        return {
            'aspect_accuracy': aspect_accuracy,
            'sentiment_accuracy': sentiment_accuracy,
            'aspect_f1': aspect_f1,
            'sentiment_f1': sentiment_f1,
            'aspect_report': aspect_report,
            'sentiment_report': sentiment_report,
            'aspect_predictions': aspect_pred,
            'sentiment_predictions': sentiment_pred
        }
    
    def save(self, model_dir: str):
        """Lưu models"""
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(self.aspect_model, os.path.join(model_dir, 'aspect_model.pkl'))
        joblib.dump(self.sentiment_model, os.path.join(model_dir, 'sentiment_model.pkl'))
    
    def load(self, model_dir: str):
        """Load models"""
        self.aspect_model = joblib.load(os.path.join(model_dir, 'aspect_model.pkl'))
        self.sentiment_model = joblib.load(os.path.join(model_dir, 'sentiment_model.pkl'))
        self.is_fitted = True