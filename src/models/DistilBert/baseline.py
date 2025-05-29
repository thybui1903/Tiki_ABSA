"""
Script dùng để phân tích trực tiếp từ dòng lệnh
"""

import sys
import json
from models.model import load_model_and_tokenizer, load_trained_model
from utils.helpers import get_device, predict_aspects, format_predictions
from config import CHECKPOINT_PATH


def baseline(input_text, checkpoint=None):
    """
    Phân tích ABSA cho văn bản đầu vào sử dụng mô hình pretrained
    
    Args:
        input_text (str): Văn bản cần phân tích
        checkpoint (str, optional): Đường dẫn đến checkpoint mô hình
        
    Returns:
        str: Kết quả phân tích dưới dạng JSON
    """
    # Tải thiết bị
    device = get_device()
    
    # Tải mô hình và tokenizer
    if checkpoint:
        model, tokenizer = load_model_and_tokenizer()
        model.load_state_dict(load_trained_model(checkpoint).state_dict())
    else:
        model, tokenizer = load_model_and_tokenizer()
    
    # Đưa mô hình lên thiết bị
    model = model.to(device)
    
    # Dự đoán
    aspect_sentiments = predict_aspects(input_text, model, tokenizer, device)
    
    # Định dạng kết quả
    return format_predictions(input_text, aspect_sentiments)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Cách sử dụng: python baseline.py <văn_bản> [đường_dẫn_checkpoint]")
        sys.exit(1)
    
    input_text = sys.argv[1]
    checkpoint = sys.argv[2] if len(sys.argv) > 2 else None
    
    result = baseline(input_text, checkpoint)
    print(result)