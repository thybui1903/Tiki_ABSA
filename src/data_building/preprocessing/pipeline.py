from src.data_building.preprocessing.text_utils import (
    normalize_whitespace,
    remove_urls,
    remove_emojis,
    remove_special_chars,
    normalize_slang,
    is_valid_comment,
)

def preprocess_pipeline(raw_comment, seen_set):
    if not raw_comment:
        return None

    text = normalize_whitespace(raw_comment)
    text = remove_urls(text)
    text = remove_emojis(text)
    text = remove_special_chars(text)
    text = normalize_slang(text)

    if not is_valid_comment(text, seen_set):
        return None

    seen_set.add(text)

    return {
        "raw_comment": raw_comment,
        "cleaned_comment": text,
    }
