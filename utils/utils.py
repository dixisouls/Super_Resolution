import logging
import os
from config import Config


def setup_logging(config: Config, log_mode):
    """Configures the logger."""

    # create the log directory if it does not exist
    os.makedirs(config.LOG_DIR, exist_ok=True)

    # get logger
    logger = logging.getLogger("super_resolution")
    logger.setLevel(config.LOG_LEVEL)

    # create file handler
    file_handler = logging.FileHandler(log_mode)
    file_handler.setLevel(config.LOG_LEVEL)

    # create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.LOG_LEVEL)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(config.LOG_FORMAT)
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    return logger
