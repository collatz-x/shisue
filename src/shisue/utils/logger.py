import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


def setup_logging(
    name: str = 'mri_scan_segmentation',
    log_dir: Optional[Path] = None,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    console: bool = True,
    file: bool = True
) -> logging.Logger:
    '''
    Setup logger with console and file handlers.

    Args:
        name: Logger name
        log_dir: Directory to save log files
        log_file: Log file name (if None, auto-generate with timestamp)
        level: Logging level
        console: Whether to log to console
        file: Whether to log to file

    Returns:
        Configured logger instance
    '''
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Remove existing handlers to avoid duplicates
    if logger.handlers:
        logger.handlers.clear()

    # Formatter
    formatter = logging.Formatter(
        fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler
    if file:
        if log_dir is None:
            log_dir = Path('logs')
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        if log_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = f'{timestamp}.log'

        file_handler = logging.FileHandler(log_dir / log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = 'mri_scan_segmentation') -> logging.Logger:
    '''
    Get existing logger or create a new one.

    Args:
        name: Logger name

    Returns:
        Logger instance
    '''
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger = setup_logging(name)
    return logger