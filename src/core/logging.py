"""
Structured logging module for Authentica AI.
"""
import logging
import sys
from typing import Optional


def get_logger(name: str = "authentica", level: Optional[str] = None) -> logging.Logger:
    """
    Returns a configured structured logger.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            fmt="%(asctime)s [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(getattr(logging, (level or "INFO").upper(), logging.INFO))
        logger.propagate = False
    return logger
