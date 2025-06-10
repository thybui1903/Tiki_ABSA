from config import Config
from utils import DataLoader, TextVectorizer
from model import ABSALogisticRegression
from visualization import ResultVisualizer
import pandas as pd
import os
import json
from sklearn.multiclass import OneVsRestClassifier

def main():
    # Khởi tạo config
    config = Config()
    
    # Tạo thư mục output
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    print("=== ASPECT-BASED SENTIMENT ANALYSIS với LOGISTIC REGRESSION ===")
    
    # 1. Load và chuẩn bị dữ liệu
    print("\n1. Loading và chuẩn bị dữ liệu...")
    data_loader = DataLoader()
    data = data_loader.prepare_data(config.TRAIN_FILE, config.VAL_FILE, config.TEST_FILE)
    
    train_texts, train_aspects, train_sentiments = data['train']
    val_texts, val_aspects, val_sentiments = data['val']
    test_texts, test_aspects, test_sentiments = data['test']
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    
    # 2. Vectorize text
    print("\n2. Vectorizing text...")
    vectorizer = TextVectorizer(
        max_features=config.MAX_FEATURES,
        ngram_range=config.NGRAM_RANGE,
        min_df=config.MIN_DF,
        max_df=config.MAX_DF
    )
    
    X_train = vectorizer.fit_transform(train_texts)
    X_val = vectorizer.transform(val_texts)
    X_test = vectorizer.transform(test_texts)
    
    print(f"Feature dimension: {X_train.shape[1]}")
    
    # 3. Training model
    print("\n3. Training model...")
    model = ABSALogisticRegression(
        C=config.C,
        max_iter=config.MAX_ITER,
        random_state=config.RANDOM_STATE
    )
    
    model.fit(X_train, train_aspects, train_sentiments)
    
    # 4. Evaluation
    print("\n4. Evaluating model...")
    
    # Train evaluation
    train_results = model.evaluate(X_train, train_aspects, train_sentiments)
    print(f"Train - Aspect Accuracy: {train_results['aspect_accuracy']:.4f}")
    print(f"Train - Sentiment Accuracy: {train_results['sentiment_accuracy']:.4f}")
    
    # Validation evaluation
    val_results = model.evaluate(X_val, val_aspects, val_sentiments)
    print(f"Validation - Aspect Accuracy: {val_results['aspect_accuracy']:.4f}")
    print(f"Validation - Sentiment Accuracy: {val_results['sentiment_accuracy']:.4f}")
    
    # Test evaluation
    test_results = model.evaluate(X_test, test_aspects, test_sentiments)
    print(f"Test - Aspect Accuracy: {test_results['aspect_accuracy']:.4f}")
    print(f"Test - Sentiment Accuracy: {test_results['sentiment_accuracy']:.4f}")
    
    # 5. Visualization
    print("\n5. Creating visualizations...")
    visualizer = ResultVisualizer(config.PLOTS_DIR)
    
    # Get label names
    aspect_labels = data_loader.aspect_encoder.classes_
    sentiment_labels = data_loader.sentiment_encoder.classes_
    
    # Plot training metrics
    visualizer.plot_training_history(train_results, val_results)
    
    # Plot confusion matrices
    visualizer.plot_confusion_matrices(
        test_aspects, test_results['aspect_predictions'],
        test_sentiments, test_results['sentiment_predictions'],
        aspect_labels, sentiment_labels
    )
    
    # Plot classification reports
    visualizer.plot_classification_report(
        test_results['aspect_report'], test_results['sentiment_report'],
        aspect_labels, sentiment_labels
    )
    
    # Error analysis
    errors_data = []
    for i, (true_asp, pred_asp, true_sent, pred_sent, text) in enumerate(zip(
        test_aspects, test_results['aspect_predictions'],
        test_sentiments, test_results['sentiment_predictions'],
        test_texts
    )):
        if true_asp != pred_asp or true_sent != pred_sent:
            errors_data.append({
                'text': text,
                'text_length': len(text.split()),
                'true_aspect': aspect_labels[true_asp],
                'pred_aspect': aspect_labels[pred_asp],
                'true_sentiment': sentiment_labels[true_sent],
                'pred_sentiment': sentiment_labels[pred_sent]
            })
    
    if errors_data:
        errors_df = pd.DataFrame(errors_data)
        visualizer.plot_error_analysis(errors_df)
    
    # 6. Save models và encoders
    print("\n6. Saving models...")
    model.save(config.MODEL_DIR)
    vectorizer.save(config.MODEL_DIR)
    data_loader.save_encoders(config.MODEL_DIR)
    
    # Save results
    results_summary = {
        # Accuracy
        'train_aspect_acc': train_results['aspect_accuracy'],
        'train_sentiment_acc': train_results['sentiment_accuracy'],
        'val_aspect_acc': val_results['aspect_accuracy'],
        'val_sentiment_acc': val_results['sentiment_accuracy'],
        'test_aspect_acc': test_results['aspect_accuracy'],
        'test_sentiment_acc': test_results['sentiment_accuracy'],

        # F1 (macro)
        'train_aspect_f1_macro': train_results['aspect_f1']['macro'],
        'train_sentiment_f1_macro': train_results['sentiment_f1']['macro'],
        'val_aspect_f1_macro': val_results['aspect_f1']['macro'],
        'val_sentiment_f1_macro': val_results['sentiment_f1']['macro'],
        'test_aspect_f1_macro': test_results['aspect_f1']['macro'],
        'test_sentiment_f1_macro': test_results['sentiment_f1']['macro'],

        # F1 (weighted)
        'train_aspect_f1_weighted': train_results['aspect_f1']['weighted'],
        'train_sentiment_f1_weighted': train_results['sentiment_f1']['weighted'],
        'val_aspect_f1_weighted': val_results['aspect_f1']['weighted'],
        'val_sentiment_f1_weighted': val_results['sentiment_f1']['weighted'],
        'test_aspect_f1_weighted': test_results['aspect_f1']['weighted'],
        'test_sentiment_f1_weighted': test_results['sentiment_f1']['weighted']
}

    
    with open(os.path.join(config.RESULTS_DIR, 'results_summary.json'), 'w') as f:
        json.dump(results_summary, f, indent=2)
    
    print("\n=== HOÀN THÀNH ===")
    print(f"Models saved to: {config.MODEL_DIR}")
    print(f"Plots saved to: {config.PLOTS_DIR}")
    print(f"Results saved to: {config.RESULTS_DIR}")

if __name__ == "__main__":
    main()