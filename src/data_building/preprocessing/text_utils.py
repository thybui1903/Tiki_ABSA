import re
import unicodedata
import emoji
from src.data_building.config import SPAM_PHRASES, MIN_WORDS_IN_COMMENT, SLANG_DICT # Import từ config

# ===== Cleaning functions =====

def normalize_whitespace(text):
    text = re.sub(r'[\r\n]+', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_urls(text):
    return re.sub(r'http\S+|https?://\S+', '', text)

def remove_emojis(text):
    return emoji.replace_emoji(text, '')

def remove_special_chars(text):
    text = unicodedata.normalize('NFC', text)
    text = re.sub(r"[^\w\s.,!?;:]", '', text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def normalize_slang(text):
    # Sử dụng SLANG_DICT từ config
    for slang, full in SLANG_DICT.items():
        text = re.sub(r'\b' + re.escape(slang) + r'\b', full, text)
    return text

# ===== Filtering functions =====

def is_spam(comment):
    lower = comment.lower()
    # Sử dụng SPAM_PHRASES từ config
    return any(lower == phrase or lower.strip('.') == phrase for phrase in SPAM_PHRASES)

def is_valid_comment(comment, seen_set):
    # Sử dụng MIN_WORDS_IN_COMMENT từ config
    return len(comment.split()) >= MIN_WORDS_IN_COMMENT and comment not in seen_set and not is_spam(comment)