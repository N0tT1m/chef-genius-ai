"""
Centralized logging configuration for ChefGenius application.
"""

import logging
import sys
from pathlib import Path
from typing import Optional
from loguru import logger
from .config import settings


class InterceptHandler(logging.Handler):
    """Intercept standard logging messages toward loguru sinks."""
    
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


def setup_logging() -> None:
    """Configure logging for the application."""
    
    # Remove default loguru handler
    logger.remove()
    
    # Configure loguru with custom format
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )
    
    # Console handler
    logger.add(
        sys.stderr,
        format=log_format,
        level=settings.LOG_LEVEL,
        colorize=True,
        backtrace=True,
        diagnose=True,
    )
    
    # File handler (if configured)
    if settings.LOG_FILE:
        log_file_path = Path(settings.LOG_FILE)
        log_file_path.parent.mkdir(parents=True, exist_ok=True)
        
        logger.add(
            log_file_path,
            format=log_format,
            level=settings.LOG_LEVEL,
            rotation=settings.LOG_ROTATION,
            retention=settings.LOG_RETENTION,
            compression="gz",
            backtrace=True,
            diagnose=True,
        )
    
    # Intercept standard logging
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
    
    # Set levels for third-party loggers
    third_party_loggers = [
        "uvicorn",
        "uvicorn.error", 
        "uvicorn.access",
        "fastapi",
        "sqlalchemy",
        "alembic",
        "elasticsearch",
        "transformers",
        "torch",
        "PIL",
    ]
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).handlers = [InterceptHandler()]
        if settings.is_development():
            logging.getLogger(logger_name).setLevel(logging.INFO)
        else:
            logging.getLogger(logger_name).setLevel(logging.WARNING)


def get_logger(name: str) -> "logger":
    """Get a logger instance with the given name."""
    return logger.bind(name=name)


# Application logger instance
app_logger = get_logger("chefgenius")