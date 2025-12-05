"""Logging utilities for agent."""
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional
import colorama
from colorama import Fore, Style

from kai.config.paths import LOGS_DIR


# Initialize colorama for cross-platform colored output
colorama.init()

# Global filter that will be applied to all loggers created via setup_logger
_GLOBAL_FILTER = None


def set_global_filter(filter_instance: logging.Filter):
    """
    Set a global filter that will be applied to all loggers created via setup_logger.

    This should be called BEFORE any loggers are created to ensure consistent filtering.

    Args:
        filter_instance: Filter to apply to all new loggers
    """
    global _GLOBAL_FILTER
    _GLOBAL_FILTER = filter_instance


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colored output."""

    COLORS = {
        'DEBUG': Fore.CYAN,
        'INFO': Fore.GREEN,
        'WARNING': Fore.YELLOW,
        'ERROR': Fore.RED,
        'CRITICAL': Fore.RED + Style.BRIGHT,
    }

    def format(self, record):
        # Add color to the level name
        levelname = record.levelname
        if levelname in self.COLORS:
            record.levelname = f"{self.COLORS[levelname]}{levelname}{Style.RESET_ALL}"

        # Format the message
        formatted = super().format(record)

        return formatted


def setup_logger(
    name: str,
    level: Optional[str] = None,
    log_file: Optional[Path] = None,
    console: bool = True,
    file_logging: bool = True,
) -> logging.Logger:
    """Set up a logger with console and file handlers.
    
    Args:
        name: Logger name (usually __name__)
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (defaults to logs/kai_agent.log)
        console: Enable console logging
        file_logging: Enable file logging
        
    Returns:
        Configured logger
    """
    # Get or create logger
    logger = logging.getLogger(name)
    
    # Set level
    if level is None:
        level = "INFO"
    logger.setLevel(getattr(logging, level.upper()))
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stderr)
        console_handler.setLevel(logging.DEBUG)

        # Use colored formatter for console
        console_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        console_formatter = ColoredFormatter(console_format, datefmt="%H:%M:%S")
        console_handler.setFormatter(console_formatter)

        # Apply global filter if set
        if _GLOBAL_FILTER is not None:
            console_handler.addFilter(_GLOBAL_FILTER)

        logger.addHandler(console_handler)
    
    # File handler
    if file_logging:
        if log_file is None:
            # Default log file location
            log_dir = LOGS_DIR
            log_dir.mkdir(parents=True, exist_ok=True)
            
            # Create timestamped log file
            timestamp = datetime.now().strftime("%Y%m%d")
            log_file = log_dir / f"kai_agent_{timestamp}.log"
        else:
            # Ensure log directory exists
            log_file.parent.mkdir(parents=True, exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Plain formatter for file (no colors)
        file_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        file_formatter = logging.Formatter(file_format)
        file_handler.setFormatter(file_formatter)
        
        logger.addHandler(file_handler)
    
    # Prevent propagation to root logger
    logger.propagate = False
    
    return logger


def get_logger(name: str) -> logging.Logger:
    """Get an existing logger.
    
    Args:
        name: Logger name
        
    Returns:
        Logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: str, logger_name: Optional[str] = None):
    """Set logging level for a logger.
    
    Args:
        level: Logging level
        logger_name: Logger name (None for root logger)
    """
    if logger_name:
        logger = logging.getLogger(logger_name)
    else:
        logger = logging.getLogger()
    
    logger.setLevel(getattr(logging, level.upper()))
    
    # Also update handlers
    for handler in logger.handlers:
        handler.setLevel(getattr(logging, level.upper()))


def log_function_call(logger: logging.Logger):
    """Decorator to log function calls.
    
    Args:
        logger: Logger to use
        
    Returns:
        Decorator function
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            logger.debug(f"Calling {func_name} with args={args}, kwargs={kwargs}")
            
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func_name} returned: {result}")
                return result
            except Exception as e:
                logger.error(f"{func_name} raised exception: {e}")
                raise
        
        return wrapper
    return decorator


def log_execution_time(logger: logging.Logger):
    """Decorator to log function execution time.
    
    Args:
        logger: Logger to use
        
    Returns:
        Decorator function
    """
    import time
    
    def decorator(func):
        def wrapper(*args, **kwargs):
            func_name = func.__name__
            start_time = time.time()
            
            try:
                result = func(*args, **kwargs)
                elapsed = time.time() - start_time
                logger.info(f"{func_name} completed in {elapsed:.2f} seconds")
                return result
            except Exception as e:
                elapsed = time.time() - start_time
                logger.error(f"{func_name} failed after {elapsed:.2f} seconds: {e}")
                raise
        
        return wrapper
    return decorator


class LogContext:
    """Context manager for temporary log level changes."""
    
    def __init__(self, logger: logging.Logger, level: str):
        """Initialize log context.
        
        Args:
            logger: Logger to modify
            level: Temporary log level
        """
        self.logger = logger
        self.new_level = getattr(logging, level.upper())
        self.old_level = logger.level
    
    def __enter__(self):
        """Enter context - set new log level."""
        self.logger.setLevel(self.new_level)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit context - restore old log level."""
        self.logger.setLevel(self.old_level)
        return False