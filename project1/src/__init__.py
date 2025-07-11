"""
Airbnb Sentiment Analysis & Satisfaction Forecasting Package

A comprehensive toolkit for analyzing customer sentiment from Airbnb reviews
and forecasting satisfaction trends using machine learning techniques.
"""

__version__ = "1.0.0"
__author__ = "Aniket Gupta"
__email__ = "your.email@example.com"

from .analysis import sentiment, satisfaction
from .forecasting import forecaster  
from .data import loader
from .utils import logging_config

__all__ = [
    'sentiment',
    'satisfaction', 
    'forecaster',
    'loader',
    'logging_config'
] 