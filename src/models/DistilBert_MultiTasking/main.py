# main.py
import torch
import os
from transformers import AutoTokenizer
import json
from config import ModelConfig, TrainingConfig
from utils import DataLoader, convert_to_json_serializable
from model import ABSADataset, ABSAPredictor, MultiTaskABSAModel
from trainer import ABSATrainer

def main():
    """Main training and evaluation pipeline"""
    
    # Initialize configurations
    model_config = ModelConfig()
    training_config = TrainingConfig()
    
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
    
    # Load best model for evaluation
    checkpoint = torch.load(
        os.path.join(training_config.model_save_path, "best_model.pt")
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Initialize predictor
    tokenizer = AutoTokenizer.from_pretrained(model_config.model_name)
    predictor = ABSAPredictor(model, tokenizer, model_config)
    
    # Test prediction
    test_text = "Giao hàng nhanh. Túi chắc đẹp. Mua lần thứ 2 rồi. Đợt này tiki giá tốt hơn so với 2 trang kia"
    prediction = predictor.predict(test_text)
    
    print("Test prediction:")
    print(json.dumps(convert_to_json_serializable(prediction), ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()