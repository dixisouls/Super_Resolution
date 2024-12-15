import os


class Config:

    # data paths
    DATA_DIR = "data"
    HIGH_RES_DIR = os.path.join(DATA_DIR, "high_res")
    LOW_RES_DIR = os.path.join(DATA_DIR, "low_res")

    # model settings
    MODEL_NAME = "edsr_with_attention"
    SCALE_FACTOR = 8
    NUM_CHANNELS = 3
    NUM_FEATURES = 64
    NUM_RES_BLOCKS = 32
    REDUCTION_RATIO = 16

    # training settings
    BATCH_SIZE = 8
    EPOCHS = 10
    LEARNING_RATE = 1e-4
    SAVE_MODEL_EVERY = 10
    OUTPUT_DIR = "trained_models"

    # inference settings
    MODEL_PATH = os.path.join(OUTPUT_DIR, f"best.pth")
    TEST_DIRECTORY = "test_images"
    OUTPUT_IMAGE_PATH = "output.png"
    OUTPUT_VISUALIZATION_PATH = "visualization.png"

    # logging settings
    LOG_DIR = "logs"
    TRAIN_LOG = os.path.join(LOG_DIR, "train.log")
    INFER_LOG = os.path.join(LOG_DIR, "infer.log")
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
