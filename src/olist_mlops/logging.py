import logging

from olist_mlops.config import PROJECT_ROOT, config


def setup_logging() -> logging.Logger:
    """Configure application logging to console and file."""

    log_file = PROJECT_ROOT / config["logging"]["file"]
    log_file.parent.mkdir(parents=True, exist_ok=True)

    level = getattr(
        logging,
        config["logging"]["level"].upper(),
        logging.INFO,
    )

    logger = logging.getLogger("olist_mlops")
    logger.setLevel(level)

    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    )

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    file_handler = logging.FileHandler(
        log_file,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    return logger
