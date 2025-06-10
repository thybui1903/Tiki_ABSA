"""
Module chứa các lớp và hàm xử lý dataset
"""

import json
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import sys
import os

# Thêm thư mục gốc vào đường dẫn
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import MAX_LENGTH, BATCH_SIZE, RANDOM_SEED, LABEL2ID, ID2LABEL


class AspectSentimentDataset(Dataset):
    """
    Dataset cho bài toán Aspect-based Sentiment Analysis
    """
    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        aspect_sentiments = self.labels[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Tạo nhãn cho mỗi token, sử dụng BIO tagging
        token_labels = torch.ones(self.max_length, dtype=torch.long) * -100  # ✅ sửa dtype
        
        for start_char, end_char, text_span, aspect_sentiment in aspect_sentiments:
            # Map vị trí ký tự sang vị trí token
            token_start = encoding.char_to_token(start_char)
            token_end = encoding.char_to_token(end_char - 1)
            
            if token_start is None or token_end is None:
                continue
                
            # Gán nhãn cho các token
            aspect_sentiment_id = LABEL2ID.get(aspect_sentiment, -100)
            token_labels[token_start:token_end+1] = aspect_sentiment_id
        
        return {
            "input_ids": encoding.input_ids.flatten(),
            "attention_mask": encoding.attention_mask.flatten(),
            "labels": token_labels
        }



def process_data(data_list):
    """
    Xử lý dữ liệu ABSA từ định dạng JSON
    
    Args:
        data_list (list): Danh sách các mẫu dữ liệu
        
    Returns:
        tuple: (texts, labels) - Danh sách các văn bản và nhãn tương ứng
    """
    texts = []
    labels = []
    
    for item in data_list:
        try:
            # Parse JSON nếu item là string
            item_dict = json.loads(item) if isinstance(item, str) else item
            
            # Kiểm tra key "text" và "labels"
            if "text" not in item_dict:
                print("Warning: Missing 'text' in item:", item_dict)
                continue
                
            if "labels" not in item_dict:
                print("Warning: Missing 'labels' in item:", item_dict)
                labels.append([])  # Thêm list rỗng nếu không có labels
                texts.append(item_dict["text"])
                continue
                
            texts.append(item_dict["text"])
            
            # Kiểm tra "labels" có phải là list không
            if not isinstance(item_dict["labels"], list):
                print("Warning: 'labels' is not a list in item:", item_dict)
                labels.append([])
                continue
                
            # Chuyển đổi định dạng nhãn
            item_labels = []
            for label in item_dict["labels"]:
                if len(label) != 4:
                    print("Warning: Invalid label format:", label)
                    continue
                start_char, end_char, text_span, aspect_sentiment = label
                item_labels.append((start_char, end_char, text_span, aspect_sentiment))
            
            labels.append(item_labels)
            
        except Exception as e:
            print("Error processing item:", item)
            print("Exception:", e)
            continue
    
    return texts, labels

def create_data_loaders(texts, labels, tokenizer, test_size=0.2):
    """
    Tạo DataLoader cho quá trình huấn luyện và đánh giá
    
    Args:
        texts (list): Danh sách các văn bản
        labels (list): Danh sách các nhãn tương ứng
        tokenizer: Tokenizer từ Hugging Face
        test_size (float): Tỷ lệ dữ liệu kiểm thử
        
    Returns:
        tuple: (train_loader, val_loader) - DataLoader cho tập huấn luyện và kiểm thử
    """
    # Chia dữ liệu thành tập train và validation
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        texts, labels, test_size=test_size, random_state=RANDOM_SEED
    )
    
    # Tạo dataset
    train_dataset = AspectSentimentDataset(train_texts, train_labels, tokenizer)
    val_dataset = AspectSentimentDataset(val_texts, val_labels, tokenizer)
    
    # Tạo dataloader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    return train_loader, val_loader