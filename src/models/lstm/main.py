from argparse import ArgumentParser
from configs.utils import get_config
from tasks.sentiment_analysis import SentimentAnalysisTask  # Tên class task chính của bạn
from datasets.sentiment_dataset import SentimentDataset

def main():
    parser = ArgumentParser()
    parser.add_argument("--config-file", type=str, required=True, help="Path to the YAML config file")
    args = parser.parse_args()

    # Load cấu hình từ file YAML
    config = get_config(args.config_file)

    # Khởi tạo task phân tích cảm xúc
    task = SentimentAnalysisTask(config)

    # Load vocab
    task.load_vocab(config["vocab"])

    # Huấn luyện / đánh giá mô hình
    task.start()

    # Lấy dự đoán (nếu có)
    _, accurancy, scores = task.get_predictions()
    print(accurancy)
    print(scores)


    # Ghi log hoàn tất
    task.logger.info("Task done!")

if __name__ == "__main__":
    main()

# python3 main.py --config-file configs/bilstm.yaml
# python3 main.py --config-file configs/lstm.yaml