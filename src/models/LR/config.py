import os

class Config:
    # Đường dẫn dữ liệu: src/clean_data
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    DATA_DIR = os.path.join(BASE_DIR, "clean_data")

    TRAIN_FILE = os.path.join(DATA_DIR, "train.json")
    VAL_FILE = os.path.join(DATA_DIR, "dev.json")
    TEST_FILE = os.path.join(DATA_DIR, "test.json")
    
    # Model config
    MAX_FEATURES = 10000
    NGRAM_RANGE = (1, 3)
    MIN_DF = 2
    MAX_DF = 0.95
    
    # Training config
    C = 1.0  # Regularization parameter
    MAX_ITER = 1000
    RANDOM_STATE = 42
    
    # Output
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")
    RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
    PLOTS_DIR = os.path.join(os.path.dirname(__file__), "visualizations")
