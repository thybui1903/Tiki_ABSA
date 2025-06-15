# Cấu hình cho việc crawl dữ liệu
HEADERS = {
    'User-Agent': 'Mozilla/5.0',
    'x-guest-token': 'QR2XwTjP6zhyurdq71mp3GxLvfMZW4kA'
}
CATEGORY_IDS = [2549, 1815,]  # danh mục cần crawl
MAX_PER_CATEGORY = 1000  # Số lượng comment mỗi danh mục
OUTPUT_FILE_NAME = "data/raw/raw_data.json"
CRAWL_PRODUCT_PAGES = 10  # Số trang sản phẩm cần crawl
CRAWL_REVIEW_PAGES_PER_PRODUCT = 20 # Số trang bình luận mỗi sản phẩm
SLEEP_TIME = 0.5 # Thời gian chờ giữa các request để tránh bị block