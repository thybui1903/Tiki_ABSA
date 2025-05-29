"""
Module chứa các hàm liên quan đến mô hình
"""

import torch
import os
import sys
from transformers import AutoTokenizer, AutoModelForTokenClassification

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MODEL_NAME, ID2LABEL, LABEL2ID


def load_model_and_tokenizer():
    """
    Tải mô hình và tokenizer
    
    Returns:
        tuple: (model, tokenizer) - Mô hình và tokenizer đã tải
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Tạo mô hình với số lượng nhãn tương ứng
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(ID2LABEL),
        id2label=ID2LABEL,
        label2id=LABEL2ID
    )
    
    return model, tokenizer


def save_model(model, path):
    """
    Lưu mô hình
    
    Args:
        model: Mô hình cần lưu
        path (str): Đường dẫn để lưu mô hình
    """
    # Tạo thư mục nếu chưa tồn tại
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Lưu mô hình
    torch.save(model.state_dict(), path)
    print(f"Đã lưu mô hình tại: {path}")


def load_trained_model(path):
    """
    Tải mô hình đã huấn luyện
    
    Args:
        path (str): Đường dẫn đến mô hình đã lưu
        
    Returns:
        model: Mô hình đã tải
    """
    model, _ = load_model_and_tokenizer()
    
    # Tải trọng số
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    
    return model


def train_model(model, train_loader, val_loader, device, epochs=5, learning_rate=5e-5, checkpoint_path=None):
    """
    Huấn luyện mô hình
    
    Args:
        model: Mô hình cần huấn luyện
        train_loader: DataLoader cho tập huấn luyện
        val_loader: DataLoader cho tập đánh giá
        device: Thiết bị chạy mô hình (CPU/GPU)
        epochs (int): Số lượng epoch
        learning_rate (float): Tốc độ học
        checkpoint_path (str): Đường dẫn để lưu mô hình tốt nhất
        
    Returns:
        model: Mô hình đã huấn luyện
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0
        
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Lưu mô hình tốt nhất
        if checkpoint_path and avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_model(model, checkpoint_path)
            print(f"Đã lưu mô hình tốt nhất với val loss: {best_val_loss:.4f}")
    
    return model