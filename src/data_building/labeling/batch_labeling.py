import json
import os
import re
import time
from src.data_building.labeling.gemini_utils import call_gemini_sentiment

def process_reviews_by_batch(input_file, output_file, batch_size=3, sleep_time=2):
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            all_data = json.load(f)

        cleaned_comments = [item.get('cleaned_comment', '') for item in all_data if item.get('cleaned_comment')]
        total_samples = len(cleaned_comments)

        print(f"📄 Tổng số review: {total_samples}")

        results = []
        if os.path.exists(output_file):
            with open(output_file, 'r', encoding='utf-8') as f:
                results = [json.loads(line) for line in f if line.strip()]

        start_index = len(results)
        print(f"⏩ Tiếp tục từ sample {start_index}")

        for i in range(start_index, total_samples):
            review = cleaned_comments[i]
            print(f"[{i+1}/{total_samples}] Đang xử lý...")

            result = call_gemini_sentiment(review)
            if result:
                cleaned_result = re.sub(r'\s+', ' ', result).strip()
                cleaned_result = cleaned_result.replace("```json", "").replace("```", "")
                try:
                    parsed_result = json.loads(cleaned_result)
                    results.append(parsed_result)
                except json.JSONDecodeError:
                    print(f"⚠️ Lỗi JSON ở sample {i + 1}, lưu chuỗi gốc")
                    results.append({"text": review, "raw_result": cleaned_result})
            else:
                results.append({"text": review, "error": "No result"})

            if (i + 1) % batch_size == 0 or i == total_samples - 1:
                with open(output_file, 'w', encoding='utf-8') as f:
                    for item in results:
                        f.write(json.dumps(item, ensure_ascii=False) + '\n')
                print(f"✅ Đã lưu {len(results)} kết quả vào '{output_file}'")

            time.sleep(sleep_time)

    except FileNotFoundError:
        print(f"🚫 Không tìm thấy file '{input_file}'")
    except Exception as e:
        print(f"❗ Đã xảy ra lỗi: {e}")