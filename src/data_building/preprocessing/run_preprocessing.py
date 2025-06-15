import json
from src.data_building.config import RAW_DATA_PATH, PROCESSED_DATA_PATH
from src.data_building.preprocessing.pipeline import preprocess_pipeline

if __name__ == "__main__":
    print("[INFO] Starting preprocessing...")

    seen_set = set()

    # Load raw data
    with open(RAW_DATA_PATH, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    processed_reviews = []

    for entry in raw_data:
        raw_comment = entry.get("content", "")
        result = preprocess_pipeline(raw_comment, seen_set)
        if result:  # Nếu không bị lọc
            processed_reviews.append(result)

    # Save
    with open(PROCESSED_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(processed_reviews, f, ensure_ascii=False, indent=2)

    print(f"[INFO] Preprocessing completed. Cleaned data saved to {PROCESSED_DATA_PATH}")