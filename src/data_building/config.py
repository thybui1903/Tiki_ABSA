# src/data_building/config.py
# Cấu hình cho việc crawl dữ liệu
HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'x-guest-token': 'QR2XwTjP6zhyurdq71mp3GxLvfMZW4kA'
}
CATEGORY_IDS = [2549, 1815,]  # danh mục cần crawl
MAX_PER_CATEGORY = 1000  # Số lượng comment mỗi danh mục
RAW_DATA_PATH = "data/raw/raw_data.json"
CRAWL_PRODUCT_PAGES = 10  # Số trang sản phẩm cần crawl
CRAWL_REVIEW_PAGES_PER_PRODUCT = 20 # Số trang bình luận mỗi sản phẩm
SLEEP_TIME = 0.5 # Thời gian chờ giữa các request để tránh bị block

# Cấu hình đường dẫn file cho Preprocessing
RAW_DATA_PATH = 'data/raw/raw_data.json' # Đường dẫn đến file dữ liệu thô (ví dụ từ bước crawl)
PROCESSED_DATA_PATH = 'data/processed/processed_data.json' # Đường dẫn đến file dữ liệu đã được tiền xử lý

# Cấu hình cho hàm filtering (is_spam) trong text_utils
SPAM_PHRASES = ["sản phẩm tốt", "giao hàng nhanh", "ok", "được", "ngon", "like", "thanks"] # Các cụm từ được coi là spam
MIN_WORDS_IN_COMMENT = 5 # Số lượng từ tối thiểu để một bình luận được coi là hợp lệ

# Cấu hình cho việc chuẩn hóa từ lóng (normalize_slang) trong text_utils
SLANG_DICT = {
    "ko": "không", "k": "không", "hok": "không",
    "dc": "được", "đc": "được", "bt": "bình thường",
    "thik": "thích", "thấy oke": "thấy ổn", "oke": "ok", "okie": "ok",
    "mik": "mình", "j": "gì", "cx": "cũng"
}

# Cấu hình cho việc label dữ liệu
DEFAULT_MODEL = "gemini-2.0-flash" # Hoặc tên model Gemini bạn đang sử dụng
BATCH_SIZE = 3 # Kích thước batch mặc định cho quá trình xử lý review
SLEEP_TIME = 2 # Thời gian chờ giữa các batch để tránh quá tải API hoặc bị block
INPUT_FILE_PROCESSED_DATA = 'data/processed/processed_data.json' # Đường dẫn file đầu vào chứa dữ liệu đã được tiền xử lý
OUTPUT_FILE_LABELED_DATA = 'data/labeled/labeled_data.json' # Đường dẫn file đầu ra chứa dữ liệu đã được gán nhãn
