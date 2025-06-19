
import json
import pandas as pd
from sklearn.model_selection import train_test_split
from src.config import LABELED_DATA_PATH, SPLIT_DATA_DIR


def load_labeled_data(filepath=LABELED_DATA_PATH):
    """
    Load dữ liệu đã gán nhãn từ file JSON hoặc CSV.
    Mỗi dòng nên là: {"text": ..., "labels": [["ASPECT", "SENTIMENT"], ...]}
    """
    if filepath.endswith(".json"):
        with open(filepath, "r", encoding="utf-8") as f:
            data = [json.loads(line) for line in f]
        return pd.DataFrame(data)
    elif filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    else:
        raise ValueError("Unsupported file format.")


def split_data(df, test_size=0.2, val_size=0.1, random_state=42):
    """
    Chia tập dữ liệu thành train/val/test.
    Trả về: train_df, val_df, test_df
    """
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=random_state)
    train_df, val_df = train_test_split(train_df, test_size=val_size, random_state=random_state)
    return train_df, val_df, test_df


def save_splits(train_df, val_df, test_df):
    """
    Lưu các file split ra thư mục `data/splits/`.
    """
    train_df.to_json(f"{SPLIT_DATA_DIR}/train.json", orient="records", lines=True, force_ascii=False)
    val_df.to_json(f"{SPLIT_DATA_DIR}/val.json", orient="records", lines=True, force_ascii=False)
    test_df.to_json(f"{SPLIT_DATA_DIR}/test.json", orient="records", lines=True, force_ascii=False)
    print("✅ Saved split data to:", SPLIT_DATA_DIR)


def load_split_data():
    """
    Load lại các file train/val/test đã lưu.
    """
    train = pd.read_json(f"{SPLIT_DATA_DIR}/train.json", lines=True)
    val = pd.read_json(f"{SPLIT_DATA_DIR}/val.json", lines=True)
    test = pd.read_json(f"{SPLIT_DATA_DIR}/test.json", lines=True)
    return train, val, test