RAW_DATA_PATH = r"D:\Tiki_ABSA\data\raw\raw_data.json"
MAX_REVIEWS_PER_PRODUCT = 6000

PROCESSED_DATA_PATH = r"D:\Tiki_ABSA\data\processed\processed_data.json"
LABELED_DATA_PATH = r"D:\Tiki_ABSA\data\labeled\labeled_data.json"

# --- Cấu hình API ---
GOOGLE_API_KEY = 'AIzaSyBkHK5F_0j1X_dAJA4WigsD7kdriX-pAao'
genai.configure(api_key=GOOGLE_API_KEY)

# --- Tham số mặc định ---
DEFAULT_MODEL = 'gemini-2.0-flash'
