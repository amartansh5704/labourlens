# shared/logger.py
from loguru import logger
import sys
import os

def setup_logger():
    """
    Configure loguru logger for the whole project
    Call this once at the start of any script
    """

    # remove default logger
    logger.remove()

    # console logger - shows INFO and above
    logger.add(
        sys.stdout,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
               "<level>{level: <8}</level> | "
               "<cyan>{name}</cyan>:<cyan>{line}</cyan> | "
               "<level>{message}</level>",
        level="INFO",
        colorize=True
    )

    # file logger - saves everything including DEBUG
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/laborlens.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level} | {name}:{line} | {message}",
        level="DEBUG",
        rotation="10 MB",     # new file when reaches 10MB
        retention="7 days",   # keep logs for 7 days
        compression="zip"     # compress old logs
    )

    return logger


# create one shared logger instance
setup_logger()