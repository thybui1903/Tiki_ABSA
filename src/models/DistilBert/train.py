"""
Script huấn luyện mô hình ABSA
"""

import argparse
import json
from data.dataset import process_data, create_data_loaders
from models.model import load_model_and_tokenizer, train_model
from utils.helpers import get_device
from config import EPOCHS, LEARNING_RATE, CHECKPOINT_PATH
from transformers import AutoTokenizer




def main(args):
    # Tải dữ liệu
    if args.data_file:
        with open(args.data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
    else:
        # Sử dụng mẫu dữ liệu
        data = [
            {
                "text": "Giao hàng nhanh. Túi chắc đẹp. Mua lần thứ 2 rồi. đợt này tiki giá tốt hơn so với 2 trang kia. có mã giảm ổn",
                "labels": [
                    [0, 14, "Giao hàng nhanh", "Dịch vụ#Tích cực"],
                    [16, 27, "Túi chắc đẹp", "Chất lượng sản phẩm#Tích cực"],
                    [46, 80, "tiki giá tốt hơn so với 2 trang kia", "Giá cả#Tích cực"],
                    [82, 93, "có mã giảm ổn", "Giá cả#Tích cực"]
                ]
            },
            # Thêm dữ liệu mẫu khác nếu cần
        ]
    
    # Xử lý dữ liệu
    texts, labels = process_data(data)
    print(f"Đã tải {len(texts)} mẫu dữ liệu.")
    
    # Tải mô hình và tokenizer
    model, tokenizer = load_model_and_tokenizer()
    print(f"Đã tải mô hình {model.__class__.__name__}.")
    
    # Tạo data loaders
    train_loader, val_loader = create_data_loaders(texts, labels, tokenizer, test_size=args.test_size)
    print(f"Chia dữ liệu thành {len(train_loader.dataset)} mẫu train và {len(val_loader.dataset)} mẫu validation.")
    
    # Xác định thiết bị
    device = get_device()
    print(f"Sử dụng thiết bị: {device}")
    
    # Đưa mô hình lên thiết bị
    model = model.to(device)
    
    # Huấn luyện mô hình
    print(f"Bắt đầu huấn luyện mô hình với {args.epochs} epochs và learning rate {args.lr}...")
    train_model(
        model, 
        train_loader, 
        val_loader, 
        device, 
        epochs=args.epochs, 
        learning_rate=args.lr,
        checkpoint_path=args.checkpoint
    )
    
    print("Huấn luyện hoàn tất!")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Huấn luyện mô hình ABSA cho dữ liệu Tiki comment")
    parser.add_argument("--data_file", type=str, help="Đường dẫn đến file dữ liệu JSON")
    parser.add_argument("--test_size", type=float, default=0.2, help="Tỷ lệ dữ liệu kiểm thử")
    parser.add_argument("--epochs", type=int, default=EPOCHS, help="Số lượng epoch")
    parser.add_argument("--lr", type=float, default=LEARNING_RATE, help="Learning rate")
    parser.add_argument("--checkpoint", type=str, default=CHECKPOINT_PATH, help="Đường dẫn để lưu mô hình tốt nhất")
    
    args = parser.parse_args()
    main(args)