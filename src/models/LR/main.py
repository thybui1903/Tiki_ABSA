# main.py - Fixed for combined aspect-sentiment labels
from config import Config
from utils import DataLoader, TextVectorizer
from model import ABSALogisticRegression
from visualization import ResultVisualizer
import pandas as pd
import os
import json

def print_detailed_results(results, dataset_name):
    """Print detailed results for a dataset"""
    print(f"\n{dataset_name} Results:")
    print(f"  Accuracy: {results['accuracy']:.4f}")
    print(f"  F1-Macro: {results['f1_macro']:.4f}")
    print(f"  F1-Weighted: {results['f1_weighted']:.4f}")
    print(f"  F1-Micro: {results['f1_micro']:.4f}")
    print(f"  Loss: {results['loss']:.4f}")
    
    # Print breakdown if available
    breakdown = results.get('breakdown')
    if breakdown:
        print(f"  Aspect Accuracy: {breakdown['aspect_accuracy']:.4f}")
        print(f"  Sentiment Accuracy: {breakdown['sentiment_accuracy']:.4f}")
        print(f"  Aspect F1-Macro: {breakdown['aspect_f1_macro']:.4f}")
        print(f"  Sentiment F1-Macro: {breakdown['sentiment_f1_macro']:.4f}")

def get_aspect_sentiment_breakdown(y_true, y_pred, label_encoder):
    """Break down results by aspect and sentiment separately"""
    # Convert numeric labels back to string labels
    true_labels = label_encoder.inverse_transform(y_true)
    pred_labels = label_encoder.inverse_transform(y_pred)
    
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
        from sklearn.metrics import accuracy_score, f1_score
        
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

def main():
    # Initialize config
    config = Config()
    
    # Create output directories
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.PLOTS_DIR, exist_ok=True)
    
    print("=== ASPECT-BASED SENTIMENT ANALYSIS with COMBINED LABELS ===")
    
    # 1. Load and prepare data
    print("\n1. Loading and preparing data...")
    data_loader = DataLoader()
    
    try:
        data = data_loader.prepare_combined_data(config.TRAIN_FILE, config.VAL_FILE, config.TEST_FILE)
    except FileNotFoundError as e:
        print(f"Error: Could not find data file: {e}")
        return
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    train_texts, train_labels = data['train']
    val_texts, val_labels = data['val']
    test_texts, test_labels = data['test']
    
    print(f"Train samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")
    print(f"Test samples: {len(test_texts)}")
    print(f"Unique labels: {len(data_loader.label_encoder.classes_)}")
    print(f"Label examples: {list(data_loader.label_encoder.classes_[:5])}")
    
    # Check if we have data
    if len(train_texts) == 0:
        print("Error: No training data found!")
        return
    
    # 2. Vectorize text
    print("\n2. Vectorizing text...")
    vectorizer = TextVectorizer(
        max_features=config.MAX_FEATURES,
        ngram_range=config.NGRAM_RANGE,
        min_df=config.MIN_DF,
        max_df=config.MAX_DF
    )
    
    try:
        X_train = vectorizer.fit_transform(train_texts)
        X_val = vectorizer.transform(val_texts)
        X_test = vectorizer.transform(test_texts)
        
        print(f"Feature dimension: {X_train.shape[1]}")
    except Exception as e:
        print(f"Error vectorizing text: {e}")
        return
    
    # 3. Training model
    print("\n3. Training model...")
    model = ABSALogisticRegression(
        C=config.C,
        max_iter=config.MAX_ITER,
        random_state=config.RANDOM_STATE
    )
    
    try:
        model.fit(X_train, train_labels)
        print(f"Model trained with {model.n_classes} classes")
    except Exception as e:
        print(f"Error training model: {e}")
        return
    
    # 4. Evaluation
    print("\n4. Evaluating model...")
    
    try:
        # Train evaluation
        train_results = model.evaluate(X_train, train_labels)
        train_breakdown = get_aspect_sentiment_breakdown(train_labels, train_results['predictions'], data_loader.label_encoder)
        train_results['breakdown'] = train_breakdown
        print_detailed_results(train_results, "Train")
        
        # Validation evaluation
        val_results = model.evaluate(X_val, val_labels)
        val_breakdown = get_aspect_sentiment_breakdown(val_labels, val_results['predictions'], data_loader.label_encoder)
        val_results['breakdown'] = val_breakdown
        print_detailed_results(val_results, "Validation")
        
        # Test evaluation
        test_results = model.evaluate(X_test, test_labels)
        test_breakdown = get_aspect_sentiment_breakdown(test_labels, test_results['predictions'], data_loader.label_encoder)
        test_results['breakdown'] = test_breakdown
        print_detailed_results(test_results, "Test")
        
    except Exception as e:
        print(f"Error evaluating model: {e}")
        return
    
    # 5. Visualization
    print("\n5. Creating visualizations...")
    try:
        visualizer = ResultVisualizer(config.PLOTS_DIR)
        
        # Get label names
        label_names = data_loader.label_encoder.classes_
        
        # Plot training metrics
        visualizer.plot_combined_training_history(train_results, val_results)
        
        # Plot confusion matrix (IMPROVED - now shows all labels and soft matrix)
        visualizer.plot_combined_confusion_matrix(
            test_labels, test_results['predictions'], label_names, 
            label_encoder=data_loader.label_encoder  # Pass the label encoder
        )
        
        # Plot classification report
        visualizer.plot_combined_classification_report(
            test_results['classification_report'], label_names
        )
        
        # Plot comprehensive metrics
        visualizer.plot_combined_comprehensive_metrics(train_results, val_results, test_results)
        
        # Label distribution analysis (NEW)
        label_stats = visualizer.plot_label_distribution_analysis(
            data_loader.label_encoder, test_labels, test_results['predictions']
        )
        
        # Error analysis
        errors_data = []
        for i, (true_label, pred_label, text) in enumerate(zip(
            test_labels, test_results['predictions'], test_texts
        )):
            if true_label != pred_label:
                true_label_name = data_loader.label_encoder.inverse_transform([true_label])[0]
                pred_label_name = data_loader.label_encoder.inverse_transform([pred_label])[0]
                
                errors_data.append({
                    'text': text,
                    'text_length': len(text.split()),
                    'true_label': true_label_name,
                    'pred_label': pred_label_name,
                    'true_aspect': true_label_name.split('#')[0] if '#' in true_label_name else 'Unknown',
                    'true_sentiment': true_label_name.split('#')[1] if '#' in true_label_name else 'Unknown',
                    'pred_aspect': pred_label_name.split('#')[0] if '#' in pred_label_name else 'Unknown',
                    'pred_sentiment': pred_label_name.split('#')[1] if '#' in pred_label_name else 'Unknown'
                })
        
        if errors_data:
            errors_df = pd.DataFrame(errors_data)
            visualizer.plot_combined_error_analysis(errors_df)
        else:
            print("No errors found in predictions!")
            
    except Exception as e:
        print(f"Error creating visualizations: {e}")
        import traceback
        traceback.print_exc()
    
    # 6. Save models and encoders
    print("\n6. Saving models...")
    try:
        model.save(config.MODEL_DIR)
        vectorizer.save(config.MODEL_DIR)
        data_loader.save_encoders(config.MODEL_DIR)
        
        # Save comprehensive results
        results_summary = {
            # Train metrics
            'train_accuracy': float(train_results['accuracy']),
            'train_f1_macro': float(train_results['f1_macro']),
            'train_f1_weighted': float(train_results['f1_weighted']),
            'train_f1_micro': float(train_results['f1_micro']),
            'train_loss': float(train_results['loss']),
            
            # Validation metrics
            'val_accuracy': float(val_results['accuracy']),
            'val_f1_macro': float(val_results['f1_macro']),
            'val_f1_weighted': float(val_results['f1_weighted']),
            'val_f1_micro': float(val_results['f1_micro']),
            'val_loss': float(val_results['loss']),
            
            # Test metrics
            'test_accuracy': float(test_results['accuracy']),
            'test_f1_macro': float(test_results['f1_macro']),
            'test_f1_weighted': float(test_results['f1_weighted']),
            'test_f1_micro': float(test_results['f1_micro']),
            'test_loss': float(test_results['loss']),
            
            # Number of classes and samples
            'n_classes': int(model.n_classes),
            'n_train_samples': len(train_texts),
            'n_val_samples': len(val_texts),
            'n_test_samples': len(test_texts)
        }
        
        # Add breakdown metrics if available
        if train_breakdown:
            results_summary.update({
                'train_aspect_accuracy': float(train_breakdown['aspect_accuracy']),
                'train_sentiment_accuracy': float(train_breakdown['sentiment_accuracy']),
                'train_aspect_f1_macro': float(train_breakdown['aspect_f1_macro']),
                'train_sentiment_f1_macro': float(train_breakdown['sentiment_f1_macro']),
            })
        
        if val_breakdown:
            results_summary.update({
                'val_aspect_accuracy': float(val_breakdown['aspect_accuracy']),
                'val_sentiment_accuracy': float(val_breakdown['sentiment_accuracy']),
                'val_aspect_f1_macro': float(val_breakdown['aspect_f1_macro']),
                'val_sentiment_f1_macro': float(val_breakdown['sentiment_f1_macro']),
            })
        
        if test_breakdown:
            results_summary.update({
                'test_aspect_accuracy': float(test_breakdown['aspect_accuracy']),
                'test_sentiment_accuracy': float(test_breakdown['sentiment_accuracy']),
                'test_aspect_f1_macro': float(test_breakdown['aspect_f1_macro']),
                'test_sentiment_f1_macro': float(test_breakdown['sentiment_f1_macro']),
            })
        
        with open(os.path.join(config.RESULTS_DIR, 'results_summary.json'), 'w', encoding='utf-8') as f:
            json.dump(results_summary, f, indent=2, ensure_ascii=False)
            
        print("Models and results saved successfully!")
        
    except Exception as e:
        print(f"Error saving models: {e}")
    
    print("\n=== TRAINING COMPLETED ===")
    print(f"Final Test Results:")
    print(f"  Overall Accuracy: {test_results['accuracy']:.4f}")
    print(f"  Macro F1-Score: {test_results['f1_macro']:.4f}")
    print(f"  Weighted F1-Score: {test_results['f1_weighted']:.4f}")
    print(f"  Micro F1-Score: {test_results['f1_micro']:.4f}")
    print(f"  Loss: {test_results['loss']:.4f}")
    
    if test_breakdown:
        print(f"  Aspect Accuracy: {test_breakdown['aspect_accuracy']:.4f}")
        print(f"  Sentiment Accuracy: {test_breakdown['sentiment_accuracy']:.4f}")

if __name__ == "__main__":
    main()