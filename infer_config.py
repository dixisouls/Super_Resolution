class Config:
    # model settings
    MODEL_NAME = "edsr_with_attention"
    SCALE_FACTOR = 8
    NUM_CHANNELS = 3
    NUM_FEATURES = 64
    NUM_RES_BLOCKS = 32
    REDUCTION_RATIO = 16

    # inference settings
    MODEL_PATH = "trained_models/best.pth"
    OUTPUT_IMAGE_PATH = "output.png"

    # logging setup
    LOG_LEVEL = "INFO"
    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
