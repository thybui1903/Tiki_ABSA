# main.py
import torch
import os
from transformers import AutoTokenizer
import json
from config import ModelConfig, TrainingConfig
from utils import DataLoader, convert_to_json_serializable
from model import ABSADataset, ABSAPredictor, MultiTaskABSAModel
from trainer import ABSATrainer
from detailed_evaluate import DetailedEvaluator

def run_training(model_config, training_config):
    """Function to handle training pipeline"""
    # Initialize data loader
    data_loader = DataLoader(model_config)
    
    # Load datasets
    print("Loading datasets...")
    train_data = data_loader.load_data(training_config.train_data_path)
    val_data = data_loader.load_data(training_config.val_data_path)
    test_data = data_loader.load_data(training_config.test_data_path)
    
    # Create datasets
    train_dataset = ABSADataset(train_data, data_loader)
    val_dataset = ABSADataset(val_data, data_loader)
    test_dataset = ABSADataset(test_data, data_loader)
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Initialize model
    model = MultiTaskABSAModel(model_config)
    print(f"Model initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Initialize trainer
    trainer = ABSATrainer(model, model_config, training_config)
    
    # Train model
    print("Starting training...")
    trainer.train(train_dataset, val_dataset)
    
    return model

def run_detailed_evaluation(model_config, training_config):
    """Function to handle detailed evaluation"""
    # Model path
    model_path = os.path.join(training_config.model_save_path, "checkpoints/best_model.pt")
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        print("Available files in model directory:")
        model_dir = os.path.dirname(model_path)
        if os.path.exists(model_dir):
            for file in os.listdir(model_dir):
                print(f"  - {file}")
        
        # Try alternative paths
        alternative_paths = [
            os.path.join(training_config.model_save_path, "best_model.pt"),
            "./saved_models/best_model.pt",
            "./checkpoints/best_model.pt"
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"Found model at alternative path: {alt_path}")
                model_path = alt_path
                break
        else:
            print("No model file found. Please train the model first.")
            return None
    
    # Check if test data exists
    if not os.path.exists(training_config.test_data_path):
        print(f"Error: Test data file not found at {training_config.test_data_path}")
        return None
    
    print(f"Using model from: {model_path}")
    print(f"Using test data from: {training_config.test_data_path}")
    
    # Initialize evaluator
    try:
        evaluator = DetailedEvaluator(model_path, model_config)
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        return None
    
    # Run comprehensive evaluation
    print("\n=== STARTING DETAILED EVALUATION ===")
    
    try:
        # Generate complete evaluation report
        results = evaluator.generate_evaluation_report(training_config.test_data_path)
        
        print("\n=== EVALUATION COMPLETED SUCCESSFULLY ===")
        print("Results saved to './evaluation_results/' folder:")
        print("  - detailed_metrics.json: Detailed metrics for each class")
        print("  - error_analysis.json: Error analysis")
        print("  - summary_report.txt: Summary report")
        print("  - aspect_confusion_matrices.png: Aspect confusion matrices")
        print("  - sentiment_confusion_matrices.png: Sentiment confusion matrices")
        print("  - combined_confusion_matrix.png: Combined confusion matrix")
        
        return results
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        return None

def run_individual_evaluations(model_config, training_config):
    """Function to run individual evaluation components"""
    model_path = os.path.join(training_config.model_save_path, "checkpoints/best_model.pt")
    
    if not os.path.exists(model_path):
        # Try alternative paths
        alternative_paths = [
            os.path.join(training_config.model_save_path, "best_model.pt"),
            "./saved_models/best_model.pt"
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        else:
            print("No model file found for individual evaluation.")
            return
    
    try:
        evaluator = DetailedEvaluator(model_path, model_config)
        
        print("\n=== RUNNING INDIVIDUAL EVALUATIONS ===")
        
        # 1. Basic metrics evaluation
        print("\n1. Computing detailed metrics...")
        results = evaluator.evaluate_dataset(training_config.test_data_path)
        
        # 2. Create confusion matrices
        print("\n2. Creating confusion matrices...")
        evaluator.create_confusion_matrices(training_config.test_data_path)
        
        # 3. Error analysis
        print("\n3. Performing error analysis...")
        error_analysis = evaluator.analyze_errors(training_config.test_data_path)
        
        # Print summary statistics
        print("\n=== SUMMARY STATISTICS ===")
        if 'aspect_metrics' in results:
            aspect_f1 = results['aspect_metrics']['overall']['macro_avg']['f1-score']
            print(f"Aspect Classification F1 (Macro): {aspect_f1:.4f}")
        
        if 'sentiment_metrics' in results:
            sentiment_f1 = results['sentiment_metrics']['overall']['macro_avg']['f1-score']
            print(f"Sentiment Classification F1 (Macro): {sentiment_f1:.4f}")
        
        if 'statistics' in error_analysis:
            stats = error_analysis['statistics']
            print(f"Total samples: {stats['total_samples']}")
            print(f"Aspect error rate: {stats['aspect_error_rate']:.2%}")
            print(f"Sentiment error rate: {stats['sentiment_error_rate']:.2%}")
        
        return results
        
    except Exception as e:
        print(f"Error in individual evaluations: {e}")
        return None

def demo_prediction(model_config, training_config):
    """Function to demonstrate model prediction"""
    model_path = os.path.join(training_config.model_save_path, "checkpoints/best_model.pt")
    
    # Try to find model file
    if not os.path.exists(model_path):
        alternative_paths = [
            os.path.join(training_config.model_save_path, "best_model.pt"),
            "./saved_models/best_model.pt"
        ]
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                model_path = alt_path
                break
        else:
            print("No model file found for demo prediction.")
            return
    
    try:
        # Load model
        model = MultiTaskABSAModel(model_config)
        checkpoint = torch.load(model_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        # Initialize predictor
        tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
        predictor = ABSAPredictor(model, tokenizer, model_config)
        
        # Demo predictions
        test_texts = [
            "Giao hàng nhanh. Túi chắc đẹp. Mua lần thứ 2 rồi. Đợt này tiki giá tốt hơn so với 2 trang kia",
            "Sản phẩm chất lượng tốt nhưng giao hàng hơi chậm",
            "Giá cả hợp lý, chất lượng tạm được",
            "Dịch vụ khách hàng rất tệ, không hài lòng"
        ]
        
        print("\n=== DEMO PREDICTIONS ===")
        for i, text in enumerate(test_texts, 1):
            print(f"\nTest {i}: {text}")
            try:
                prediction = predictor.predict(text)
                print("Prediction:")
                print(json.dumps(convert_to_json_serializable(prediction), 
                                ensure_ascii=False, indent=2))
            except Exception as e:
                print(f"Error in prediction: {e}")
        
    except Exception as e:
        print(f"Error in demo prediction: {e}")

def get_config_info(model_config, training_config):
    """Safely get configuration information with error handling"""
    config_info = {
        'model_name': getattr(model_config, 'model_name', 'N/A'),
        'max_length': getattr(model_config, 'max_length', 'N/A'),
        'batch_size': getattr(training_config, 'batch_size', getattr(training_config, 'train_batch_size', 'N/A')),
        'learning_rate': getattr(training_config, 'learning_rate', 'N/A'),
        'num_epochs': getattr(training_config, 'num_epochs', getattr(training_config, 'epochs', 'N/A'))
    }
    return config_info

def main_evaluate():
    """Main function for evaluation only - moved from detailed_evaluate.py"""
    
    # Initialize configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Model path
    model_path = os.path.join(training_config.model_save_path, "best_model.pt")
    
    # Test data path
    test_data_path = training_config.test_data_path
    
    # Check if model file exists
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        
        # Try alternative paths
        alternative_paths = [
            os.path.join(training_config.model_save_path, "checkpoints/best_model.pt"),
            "./saved_models/best_model.pt",
            "./checkpoints/best_model.pt"
        ]
        
        for alt_path in alternative_paths:
            if os.path.exists(alt_path):
                print(f"Found model at alternative path: {alt_path}")
                model_path = alt_path
                break
        else:
            print("No model file found. Available files:")
            for path in alternative_paths:
                if os.path.exists(os.path.dirname(path)):
                    print(f"  Directory {os.path.dirname(path)}:")
                    for file in os.listdir(os.path.dirname(path)):
                        print(f"    - {file}")
            return
    
    if not os.path.exists(test_data_path):
        print(f"Error: Test data file not found at {test_data_path}")
        return
    
    print(f"Using model: {model_path}")
    print(f"Using test data: {test_data_path}")
    
    # Initialize evaluator
    try:
        evaluator = DetailedEvaluator(model_path, model_config)
    except Exception as e:
        print(f"Error initializing evaluator: {e}")
        return
    
    # Run comprehensive evaluation
    try:
        results = evaluator.generate_evaluation_report(test_data_path)
        
        print("\n=== EVALUATION COMPLETED ===")
        print("Check './evaluation_results/' folder for detailed results:")
        print("  - detailed_metrics.json: Chi tiết metrics từng class")
        print("  - error_analysis.json: Phân tích lỗi")
        print("  - summary_report.txt: Báo cáo tóm tắt")
        print("  - confusion_matrices.png: Confusion matrices")
        
    except Exception as e:
        print(f"Error during evaluation: {e}")

def main():
    """Main function - runs complete pipeline automatically"""
    print("=== ABSA MODEL COMPLETE TRAINING AND EVALUATION PIPELINE ===\n")
    
    # Initialize configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
    # Display configuration with error handling
    config_info = get_config_info(model_config, training_config)
    print(f"Model: {config_info['model_name']}")
    print(f"Max sequence length: {config_info['max_length']}")
    print(f"Batch size: {config_info['batch_size']}")
    print(f"Learning rate: {config_info['learning_rate']}")
    print(f"Epochs: {config_info['num_epochs']}")
    print()
    
    # Check what to run based on existing files
    model_exists = any([
        os.path.exists(os.path.join(training_config.model_save_path, "best_model.pt")),
        os.path.exists(os.path.join(training_config.model_save_path, "checkpoints/best_model.pt")),
        os.path.exists("./saved_models/best_model.pt"),
        os.path.exists("./checkpoints/best_model.pt")
    ])
    
    if model_exists:
        print("✓ Model file found. Skipping training and running evaluation only.")
        print("=" * 60)
        print("RUNNING EVALUATION ONLY")
        print("=" * 60)
        main_evaluate()
    else:
        print("No model file found. Running complete training + evaluation pipeline.")
        
        # STEP 1: TRAINING
        print("=" * 60)
        print("STEP 1: MODEL TRAINING")
        print("=" * 60)
        
        model = run_training(model_config, training_config)
        if model is None:
            print("Training failed. Exiting pipeline.")
            return
        
        print("✓ Training completed successfully!")
        
        # STEP 2: COMPREHENSIVE EVALUATION
        print("\n" + "=" * 60)
        print("STEP 2: COMPREHENSIVE EVALUATION")
        print("=" * 60)
        
        results = run_detailed_evaluation(model_config, training_config)
        if results is None:
            print("✗ Comprehensive evaluation failed.")
        else:
            print("✓ Comprehensive evaluation completed!")
        
        # STEP 3: INDIVIDUAL EVALUATION COMPONENTS
        print("\n" + "=" * 60)
        print("STEP 3: INDIVIDUAL EVALUATION COMPONENTS")
        print("=" * 60)
        
        individual_results = run_individual_evaluations(model_config, training_config)
        if individual_results is None:
            print("✗ Individual evaluations failed.")
        else:
            print("✓ Individual evaluations completed!")
        
        # STEP 4: DEMO PREDICTIONS
        print("\n" + "=" * 60)
        print("STEP 4: DEMO PREDICTIONS")
        print("=" * 60)
        
        demo_prediction(model_config, training_config)
        print("✓ Demo predictions completed!")
        
        # FINAL SUMMARY
        print("\n" + "=" * 60)
        print("COMPLETE PIPELINE FINISHED SUCCESSFULLY!")
        print("=" * 60)
        print("📁 Results saved to:")
        print("   - ./evaluation_results/: All evaluation files")
        print("     • detailed_metrics.json: Metrics for each class")
        print("     • error_analysis.json: Error analysis with examples")
        print("     • summary_report.txt: Human-readable summary")
        print("     • aspect_confusion_matrices.png: Aspect confusion matrices")
        print("     • sentiment_confusion_matrices.png: Sentiment confusion matrices")
        print("     • combined_confusion_matrix.png: Combined confusion matrix")
        print("   - ./saved_models/: Model checkpoints")
        print("   - ./logs/: Training logs")
        print("\n🎉 Full ABSA pipeline completed successfully!")

if __name__ == "__main__":
    import sys
    
    # Check command line arguments
    if len(sys.argv) > 1:
        if sys.argv[1] == "evaluate" or sys.argv[1] == "eval":
            # Run evaluation only
            print("Running evaluation only...")
            main_evaluate()
        elif sys.argv[1] == "train":
            # Run training only
            print("Running training only...")
            model_config = ModelConfig()
            training_config = TrainingConfig()
            run_training(model_config, training_config)
        elif sys.argv[1] == "predict" or sys.argv[1] == "demo":
            # Run demo prediction only
            print("Running demo prediction...")
            model_config = ModelConfig()
            training_config = TrainingConfig()
            demo_prediction(model_config, training_config)
        else:
            print("Unknown argument. Available options:")
            print("  python main.py evaluate  # Run evaluation only")
            print("  python main.py train     # Run training only")
            print("  python main.py predict   # Run demo prediction only")
            print("  python main.py           # Run complete pipeline")
    else:
        # Run complete pipeline
        main()
