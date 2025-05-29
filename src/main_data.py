import os
from config import LABELED_DATA_PATH, SPLIT_DATA_DIR
from data_loader import load_labeled_data, split_data, save_splits

def main():
    # Tạo thư mục nếu chưa có
    os.makedirs(SPLIT_DATA_DIR, exist_ok=True)

    # 1. Load dữ liệu đã gán nhãn
    df = load_labeled_data(filepath=LABELED_DATA_PATH)
    print(f"✅ Loaded {len(df)} labeled samples from {LABELED_DATA_PATH}")

    # 2. Chia dữ liệu thành train/val/test
    train_df, val_df, test_df = split_data(df)
    print(f"✅ Data split: {len(train_df)} train | {len(val_df)} val | {len(test_df)} test")

    # 3. Lưu kết quả chia
    save_splits(train_df, val_df, test_df)

if __name__ == "__main__":
    main()
