"""
Module chứa các hàm tiện ích
"""

import torch
import sys
import os

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import ID2LABEL


def get_device():
    """
    Xác định thiết bị chạy mô hình
    
    Returns:
        device: Thiết bị CPU hoặc GPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def predict_aspects(text, model, tokenizer, device):
    """
    Dự đoán các khía cạnh và cảm xúc từ văn bản
    
    Args:
        text (str): Văn bản cần phân tích
        model: Mô hình đã huấn luyện
        tokenizer: Tokenizer tương ứng với mô hình
        device: Thiết bị chạy mô hình
        
    Returns:
        list: Danh sách các khía cạnh và cảm xúc được tìm thấy
    """
    model.eval()
    model = model.to(device)
    
    # Tokenize
    encoding = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)
    
    # Dự đoán
    with torch.no_grad():
        outputs = model(**encoding)
    
    # Lấy nhãn có xác suất cao nhất cho mỗi token
    predictions = torch.argmax(outputs.logits, dim=2)
    predictions = predictions[0].cpu().numpy()
    
    # Chuyển đổi từ token ids sang dự đoán nhãn
    token_predictions = []
    for token_idx, pred in enumerate(predictions):
        if pred >= 0 and encoding.attention_mask[0][token_idx] == 1:
            token_predictions.append((token_idx, ID2LABEL.get(pred.item(), "O")))
    
    # Nhóm các token liên tiếp có cùng nhãn
    aspect_sentiments = []
    current_aspect = None
    start_token = None
    
    for token_idx, aspect in token_predictions:
        if aspect != "O":
            if current_aspect is None or current_aspect != aspect:
                if current_aspect is not None:
                    # Kết thúc nhóm hiện tại
                    start_char = encoding.token_to_chars(start_token).start
                    end_char = encoding.token_to_chars(token_idx - 1).end
                    aspect_text = text[start_char:end_char]
                    aspect_sentiments.append((start_char, end_char, aspect_text, current_aspect))
                
                # Bắt đầu nhóm mới
                current_aspect = aspect
                start_token = token_idx
        elif current_aspect is not None:
            # Kết thúc nhóm nếu gặp token "O"
            start_char = encoding.token_to_chars(start_token).start
            end_char = encoding.token_to_chars(token_idx - 1).end
            aspect_text = text[start_char:end_char]
            aspect_sentiments.append((start_char, end_char, aspect_text, current_aspect))
            current_aspect = None
    
    # Thêm nhóm cuối cùng nếu có
    if current_aspect is not None and token_predictions:
        start_char = encoding.token_to_chars(start_token).start
        end_token = token_predictions[-1][0]
        end_char = encoding.token_to_chars(end_token).end
        aspect_text = text[start_char:end_char]
        aspect_sentiments.append((start_char, end_char, aspect_text, current_aspect))
    
    return aspect_sentiments


def format_predictions(text, aspect_sentiments):
    """
    Định dạng kết quả dự đoán thành chuỗi JSON
    
    Args:
        text (str): Văn bản đầu vào
        aspect_sentiments (list): Kết quả dự đoán
        
    Returns:
        str: Chuỗi JSON chứa kết quả
    """
    import json
    
    result = {
        "text": text,
        "labels": []
    }
    
    for start, end, text_span, aspect_sentiment in aspect_sentiments:
        result["labels"].append([start, end, text_span, aspect_sentiment])
    
    return json.dumps(result, ensure_ascii=False, indent=2)


def print_predictions(text, aspect_sentiments):
    """
    In kết quả dự đoán
    
    Args:
        text (str): Văn bản đầu vào
        aspect_sentiments (list): Kết quả dự đoán
    """
    print("Text:", text)
    print("Predicted aspect-sentiments:")
    for start, end, text_span, aspect_sentiment in aspect_sentiments:
        print(f"  {text_span} ({start}:{end}) -> {aspect_sentiment}")