"""
Logging configuration for the Airbnb Sentiment Analysis project.

This module provides a centralized logging configuration that creates
both console and file handlers for comprehensive logging.
"""

import logging
import logging.config
import os
from pathlib import Path
from datetime import datetime

def setup_logging(log_level='INFO', log_dir='outputs/logs'):
    """
    Set up logging configuration for the application.
    
    Args:
        log_level (str): The logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_dir (str): Directory to store log files
    """
    # Create log directory if it doesn't exist
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_filename = f"airbnb_analysis_{timestamp}.log"
    log_filepath = os.path.join(log_dir, log_filename)
    
    # Logging configuration
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'detailed': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            },
            'simple': {
                'format': '%(asctime)s - %(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': log_level,
                'formatter': 'simple',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'detailed',
                'filename': log_filepath,
                'mode': 'w'
            }
        },
        'loggers': {
            '': {  # root logger
                'level': 'DEBUG',
                'handlers': ['console', 'file']
            }
        }
    }
    
    logging.config.dictConfig(logging_config)
    
    # Create a logger for this module
    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialized. Log file: {log_filepath}")
    
    return logger

def get_logger(name):
    """Get a logger instance with the specified name."""
    return logging.getLogger(name) 