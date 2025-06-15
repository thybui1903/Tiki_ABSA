import re
import unicodedata
import emoji

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
    slang_dict = {
        "ko": "không", "k": "không", "hok": "không",
        "dc": "được", "đc": "được", "bt": "bình thường",
        "thik": "thích", "thấy oke": "thấy ổn", "oke": "ok", "okie": "ok",
        "mik": "mình", "j": "gì", "cx": "cũng"
    }
    for slang, full in slang_dict.items():
        text = re.sub(r'\b' + re.escape(slang) + r'\b', full, text)
    return text

# ===== Filtering functions =====

def is_spam(comment):
    lower = comment.lower()
    spam_phrases = ["sản phẩm tốt", "giao hàng nhanh", "ok", "được", "ngon", "like", "thanks"]
    return any(lower == phrase or lower.strip('.') == phrase for phrase in spam_phrases)

def is_valid_comment(comment, seen_set):
    return len(comment.split()) >= 5 and comment not in seen_set and not is_spam(comment)
