"""
Simple logging utility.

Provides consistent logging across the application.
"""

import logging
from datetime import datetime
from pathlib import Path


def setup_logger(
    name: str = "FaceDetection",
    log_level: str = "INFO",
    log_file: bool = False,
) -> logging.Logger:
    """
    Set up and return a logger instance.

    Args:
        name: Logger name.
        log_level: Log level ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL").
        log_file: If True, also log to file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, log_level))

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(getattr(logging, log_level))

    # Formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handler (optional)
    if log_file:
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_handler = logging.FileHandler(
            log_dir / f"detection_{timestamp}.log"
        )
        file_handler.setLevel(getattr(logging, log_level))
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
