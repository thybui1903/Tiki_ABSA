# inference.py
import os
import pickle
from utils import DataLoader, preprocess_text

def predict_new_text(text: str, model_dir: str):
    """Predict combined aspect-sentiment for new text"""
    
    # Load vectorizer and encoders
    with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
        
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
        
    # Load model
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
        
    # Preprocess text
    processed_text = preprocess_text(text)
        
    # Vectorize
    X = vectorizer.transform([processed_text])
        
    # Predict
    pred = model.predict(X)[0]
    proba = model.predict_proba(X)[0]
        
    # Decode result
    combined_label = label_encoder.inverse_transform([pred])[0]
    
    # Split combined label back to aspect and sentiment
    if '#' in combined_label:
        aspect, sentiment = combined_label.split('#', 1)
    else:
        aspect, sentiment = combined_label, 'Unknown'
        
    return {
        'text': text,
        'combined_label': combined_label,
        'aspect': aspect,
        'sentiment': sentiment,
        'confidence': max(proba),
        'all_probabilities': {
            label_encoder.inverse_transform([i])[0]: prob 
            for i, prob in enumerate(proba)
        }
    }

def predict_batch(texts: list, model_dir: str):
    """Predict for a batch of texts"""
    
    # Load components
    with open(os.path.join(model_dir, 'vectorizer.pkl'), 'rb') as f:
        vectorizer = pickle.load(f)
        
    with open(os.path.join(model_dir, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)
        
    with open(os.path.join(model_dir, 'model.pkl'), 'rb') as f:
        model = pickle.load(f)
    
    # Preprocess texts
    processed_texts = [preprocess_text(text) for text in texts]
    
    # Vectorize
    X = vectorizer.transform(processed_texts)
    
    # Predict
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    results = []
    for i, (text, pred, proba) in enumerate(zip(texts, predictions, probabilities)):
        combined_label = label_encoder.inverse_transform([pred])[0]
        
        if '#' in combined_label:
            aspect, sentiment = combined_label.split('#', 1)
        else:
            aspect, sentiment = combined_label, 'Unknown'
            
        results.append({
            'text': text,
            'combined_label': combined_label,
            'aspect': aspect,
            'sentiment': sentiment,
            'confidence': max(proba)
        })
    
    return results

# Example usage
if __name__ == "__main__":
    # Single prediction
    sample_text = "Giao hàng nhanh. Túi chắc đẹp. Mua lần thứ 2 rồi."
    model_directory = "checkpoints"  # Adjust path as needed
    
    try:
        result = predict_new_text(sample_text, model_directory)
        print("=== SINGLE PREDICTION ===")
        print(f"Text: {result['text']}")
        print(f"Combined Label: {result['combined_label']}")
        print(f"Aspect: {result['aspect']}")
        print(f"Sentiment: {result['sentiment']}")
        print(f"Confidence: {result['confidence']:.3f}")
        
        print("\nTop 3 predictions:")
        sorted_probs = sorted(result['all_probabilities'].items(), key=lambda x: x[1], reverse=True)
        for label, prob in sorted_probs[:3]:
            print(f"  {label}: {prob:.3f}")
            
    except FileNotFoundError as e:
        print(f"Error: Could not find model files. Make sure you've trained the model first.")
        print(f"Expected directory: {model_directory}")
        print(f"Error: {e}")
    except Exception as e:
        print(f"Error during prediction: {e}")
    
    # Batch prediction example
    sample_texts = [
        "Sản phẩm chất lượng tốt, đóng gói cẩn thận",
        "Giao hàng chậm quá, không hài lòng",
        "Giá cả hợp lý, sẽ mua lại"
    ]
    
    try:
        print("\n=== BATCH PREDICTION ===")
        batch_results = predict_batch(sample_texts, model_directory)
        for i, result in enumerate(batch_results, 1):
            print(f"{i}. Text: {result['text']}")
            print(f"   → {result['aspect']} | {result['sentiment']} (conf: {result['confidence']:.3f})")
            
    except Exception as e:
        print(f"Error during batch prediction: {e}")