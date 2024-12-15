import logging
from infer_config import Config

def setup_logging(config: Config):
    """Configures the logger."""
    # get logger
    logger = logging.getLogger("super_resolution")
    logger.setLevel(config.LOG_LEVEL)

    # create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(config.LOG_LEVEL)

    # create formatter and add it to the handlers
    formatter = logging.Formatter(config.LOG_FORMAT)
    console_handler.setFormatter(formatter)

    # add the handlers to the logger
    logger.addHandler(console_handler)

    return logger