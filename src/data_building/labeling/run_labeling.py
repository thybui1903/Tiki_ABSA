from src.data_building.labeling.batch_labeling import process_reviews_by_batch
from src.data_building.config import INPUT_FILE_PROCESSED_DATA, OUTPUT_FILE_LABELED_DATA, BATCH_SIZE, SLEEP_TIME

if __name__ == "__main__":
    process_reviews_by_batch(
        input_file= INPUT_FILE_PROCESSED_DATA,
        output_file= OUTPUT_FILE_LABELED_DATA,
        batch_size= BATCH_SIZE,
        sleep_time= SLEEP_TIME
    )