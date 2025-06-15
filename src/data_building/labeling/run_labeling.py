from src.data_building.labeling.batch_labeling import process_reviews_by_batch

if __name__ == "__main__":
    process_reviews_by_batch(
        input_file='data/processed/processed_data.json',
        output_file='data/labeled/labeled_data.json',
        batch_size=3,
        sleep_time=2
    )