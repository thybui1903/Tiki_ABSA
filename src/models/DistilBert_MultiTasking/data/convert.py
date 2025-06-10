import json

# Bước 1: Đọc từng dòng trong file JSON Lines
input_file = "C:/Users/DELL/Tiki_ABSA/src/models/DistilBert_MultiTasking/data/data_split/val.json"
output_file = "C:/Users/DELL/Tiki_ABSA/src/models/DistilBert_MultiTasking/data/data_split/v2_val.json"

with open(input_file, "r", encoding="utf-8") as f_in:
    data = [json.loads(line) for line in f_in if line.strip()]

# Bước 2: Ghi lại thành JSON chuẩn (dùng list)
with open(output_file, "w", encoding="utf-8") as f_out:
    json.dump(data, f_out, ensure_ascii=False, indent=2)

print(f"✅ Đã lưu dữ liệu hợp lệ JSON vào file: {output_file}")

