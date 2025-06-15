import re
import json
from src.data_building.config import DEFAULT_MODEL
import google.generativeai as genai

def call_gemini_sentiment(review_text):
    prompt = f"""Bạn là một trợ lý phân tích cảm xúc chuyên nghiệp. Hãy đọc kỹ đánh giá sau và phân tích nó dựa trên ngữ cảnh toàn bộ câu.

**Yêu cầu:**
1. Trích xuất các cụm từ cảm xúc và gán nhãn theo các khía cạnh như: Dịch vụ, Chất lượng sản phẩm, Giá cả.
2. Trả kết quả ở định dạng JSON như:
{{
  "text": "{review_text}",
  "labels": [[confidence, start_idx, end_idx, "câu phân tích", "Chất lượng#Tích cực"]]
}}

Nếu không xác định được khía cạnh, trả về: [[confidence, start_idx, end_idx, "câu phân tích", "Khác"]]

Trong đó:
- `confidence`: float, 2 chữ số thập phân
- `start_idx`, `end_idx`: chỉ số ký tự trong câu gốc
- `"ASPECT#SENTIMENT"`: kết hợp khía cạnh và cảm xúc

**Lưu ý:**
- Đánh giá ngữ cảnh toàn câu
- Không dùng từ đơn lẻ
- Trả về JSON thuần, không giải thích thêm

**Đánh giá:** {review_text}
"""
    try:
        model = genai.GenerativeModel(DEFAULT_MODEL)
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"❌ Lỗi gọi Gemini: {e}")
        return None