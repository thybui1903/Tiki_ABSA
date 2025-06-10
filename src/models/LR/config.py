import os

class Config:
    # Đường dẫn dữ liệu
    DATA_DIR = "data"
    TRAIN_FILE = os.path.join(DATA_DIR, "C:/Users/DELL/Tiki_ABSA/src/models/LR/data/data_split/v2_train.json")
    VAL_FILE = os.path.join(DATA_DIR, "C:/Users/DELL/Tiki_ABSA/src/models/LR/data/data_split/v2_val.json")
    TEST_FILE = os.path.join(DATA_DIR, "C:/Users/DELL/Tiki_ABSA/src/models/LR/data/data_split/v2_test.json")
    
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
    MODEL_DIR = "C:/Users/DELL/Tiki_ABSA/src/models/LR/checkpoints"
    RESULTS_DIR = "C:/Users/DELL/Tiki_ABSA/src/models/LR/results"
    PLOTS_DIR = "C:/Users/DELL/Tiki_ABSA/src/models/LR/visualizations"