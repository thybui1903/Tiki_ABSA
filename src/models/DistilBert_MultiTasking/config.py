# config.py
import os
from dataclasses import dataclass
from typing import Dict, List, Optional

@dataclass
class ModelConfig:
    """Configuration for the ABSA model"""
    model_name: str = "distilbert-base-multilingual-cased"
    max_length: int = 256
    batch_size: int = 8
    learning_rate: float = 2e-5
    num_epochs: int = 15
    warmup_steps: int = 500
    weight_decay: float = 0.01
    dropout_rate: float = 0.1
    
    # Multi-task weightsS
    aspect_weight: float = 1.0
    sentiment_weight: float = 1.0
    
    # Labels
    aspect_labels: List[str] = None
    sentiment_labels: List[str] = None
    
    def __post_init__(self):
        if self.aspect_labels is None:
            self.aspect_labels = [
                "Dịch vụ", "Chất lượng sản phẩm", "Giá cả", 
                "Khác"
            ]
        if self.sentiment_labels is None:
            self.sentiment_labels = ["Tiêu cực", "Bình thường", "Tích cực"]

@dataclass
class TrainingConfig:
    """Training configuration"""
    output_dir: str = "output"
    model_save_path: str = "checkpoints"
    log_dir: str = "logs"
    save_steps: int = 500
    eval_steps: int = 500
    logging_steps: int = 100
    early_stopping_patience: int = 3
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    # ✅ Đường dẫn chính xác đến tập dữ liệu nằm ngoài src/
    train_data_path: str = "../../../data/clean_data/train.json"
    val_data_path: str = "../../../data/clean_data/dev.json"
    test_data_path: str = "../../../data/clean_data/test.json"

