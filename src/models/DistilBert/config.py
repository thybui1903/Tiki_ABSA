"""
Cấu hình dự án ABSA sử dụng DistilBERT
"""

from collections import OrderedDict

# Model
MODEL_NAME = "distilbert-base-multilingual-cased"  # Đổi lại model generic
MAX_LENGTH = 128
BATCH_SIZE = 8
LEARNING_RATE = 5e-5
EPOCHS = 5
RANDOM_SEED = 42
CHECKPOINT_PATH = "checkpoints/best_absa_model.pt"

# Aspect-Sentiment Mapping
ASPECTS = ["Dịch vụ", "Chất lượng sản phẩm", "Giá cả", "Đóng gói", "Vận chuyển", "Khác"]
SENTIMENTS = ["Tích cực", "Tiêu cực", "Bình thường"]

# Tạo nhãn tự động
ASPECT_CATEGORIES = [
    f"{aspect}#{sentiment}" 
    for aspect in ASPECTS 
    for sentiment in SENTIMENTS
]

# Kiểm tra nhãn trùng lặp
if len(ASPECT_CATEGORIES) != len(set(ASPECT_CATEGORIES)):
    raise ValueError("Phát hiện nhãn trùng lặp!")

# Ánh xạ nhãn
LABEL2ID = OrderedDict((label, idx) for idx, label in enumerate(ASPECT_CATEGORIES))
ID2LABEL = OrderedDict((idx, label) for idx, label in enumerate(ASPECT_CATEGORIES))

# In thông tin kiểm tra
print(f"⚙ Cấu hình được khởi tạo với {len(ID2LABEL)} nhãn:")
for idx, label in ID2LABEL.items():
    print(f"  {idx}: {label}")