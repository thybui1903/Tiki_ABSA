import os
from utils import DataLoader
def predict_new_text(text: str, model_dir: str):
    """Predict aspect và sentiment cho text mới"""
    import pickle
    import joblib
    
    # Load models và encoders
    with open(os.path.join(model_dir, 'C:/Users\DELL/Tiki_ABSA/src/models/LR/checkpoints/vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
    
    with open(os.path.join(model_dir, 'C:/Users\DELL/Tiki_ABSA/src/models/LR/checkpoints/aspect_encoder.pkl'), 'rb') as f:
        aspect_encoder = pickle.load(f)
        
    with open(os.path.join(model_dir, 'C:/Users\DELL/Tiki_ABSA/src/models/LR/checkpoints/sentiment_encoder.pkl'), 'rb') as f:
        sentiment_encoder = pickle.load(f)
    
    aspect_model = joblib.load(os.path.join(model_dir, 'C:/Users\DELL/Tiki_ABSA/src/models/LR/checkpoints/aspect_model.pkl'))
    sentiment_model = joblib.load(os.path.join(model_dir, 'C:/Users\DELL/Tiki_ABSA/src/models/LR/checkpoints/sentiment_model.pkl'))
    
    # Preprocess text
    data_loader = DataLoader()
    processed_text = data_loader.preprocess_text(text)
    
    # Vectorize
    X = vectorizer.transform([processed_text])
    
    # Predict
    aspect_pred = aspect_model.predict(X)[0]
    sentiment_pred = sentiment_model.predict(X)[0]
    
    # Get probabilities
    aspect_proba = aspect_model.predict_proba(X)[0]
    sentiment_proba = sentiment_model.predict_proba(X)[0]
    
    # Decode predictions
    aspect_name = aspect_encoder.inverse_transform([aspect_pred])[0]
    sentiment_name = sentiment_encoder.inverse_transform([sentiment_pred])[0]
    
    return {
        'text': text,
        'aspect': aspect_name,
        'sentiment': sentiment_name,
        'aspect_confidence': max(aspect_proba),
        'sentiment_confidence': max(sentiment_proba)
    }

# Ví dụ sử dụng inference
if __name__ == "__main__":
    # Test inference
    sample_text = "Giao hàng nhanh. Túi chắc đẹp. Mua lần thứ 2 rồi."
    result = predict_new_text(sample_text, "models")
    print(f"Text: {result['text']}")
    print(f"Aspect: {result['aspect']} (confidence: {result['aspect_confidence']:.3f})")
    print(f"Sentiment: {result['sentiment']} (confidence: {result['sentiment_confidence']:.3f})")
