"""
Visualization module for the Airbnb Sentiment Analysis project.

This module provides a centralized plotting functionality for generating
various charts and visualizations used in the analysis.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

class PlotGenerator:
    """Class for generating various plots and visualizations."""
    
    def __init__(self, style="whitegrid", figure_size=(10, 6), font_size=12, dpi=300):
        """
        Initialize the plot generator with styling options.
        
        Args:
            style: Seaborn style theme
            figure_size: Default figure size as (width, height)
            font_size: Default font size for plots
            dpi: Resolution for saved plots
        """
        self.style = style
        self.figure_size = figure_size
        self.font_size = font_size
        self.dpi = dpi
        
        # Set plotting style
        sns.set_theme(style=self.style)
        plt.rcParams['figure.figsize'] = self.figure_size
        plt.rcParams['font.size'] = self.font_size
        
    def sentiment_distribution(self, sentiment_data: pd.DataFrame, save_path: str) -> None:
        """
        Plot distribution of sentiment scores.
        
        Args:
            sentiment_data: DataFrame containing sentiment scores
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figure_size)
        sns.histplot(data=sentiment_data, x='compound_score', bins=30, kde=True)
        plt.title('Distribution of Review Sentiment Scores', fontweight='bold', pad=20)
        plt.xlabel('Sentiment Score (-1: Negative, +1: Positive)')
        plt.ylabel('Count')
        
        # Add statistics text
        mean_score = sentiment_data['compound_score'].mean()
        median_score = sentiment_data['compound_score'].median()
        plt.axvline(mean_score, color='red', linestyle='--', alpha=0.7, label=f'Mean: {mean_score:.3f}')
        plt.axvline(median_score, color='orange', linestyle='--', alpha=0.7, label=f'Median: {median_score:.3f}')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved sentiment distribution plot to {save_path}")
        
    def sentiment_by_neighborhood(self, sentiment_data: pd.DataFrame, save_path: str) -> None:
        """
        Plot sentiment scores by neighborhood.
        
        Args:
            sentiment_data: DataFrame containing sentiment scores and neighborhoods
            save_path: Path to save the plot
        """
        plt.figure(figsize=(14, 8))
        
        # Calculate neighborhood statistics for sorting
        neighborhood_stats = sentiment_data.groupby('neighborhood')['compound_score'].agg(['mean', 'count']).reset_index()
        # Filter neighborhoods with at least 5 reviews for better representation
        neighborhood_stats = neighborhood_stats[neighborhood_stats['count'] >= 5]
        top_neighborhoods = neighborhood_stats.nlargest(15, 'mean')['neighborhood'].tolist()
        
        filtered_data = sentiment_data[sentiment_data['neighborhood'].isin(top_neighborhoods)]
        
        sns.boxplot(data=filtered_data, x='neighborhood', y='compound_score', palette='viridis')
        plt.xticks(rotation=45, ha='right')
        plt.title('Sentiment Scores by Neighborhood (Top 15)', fontweight='bold', pad=20)
        plt.xlabel('Neighborhood')
        plt.ylabel('Sentiment Score')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved neighborhood sentiment plot to {save_path}")
        
    def sentiment_trends(self, sentiment_data: pd.DataFrame, save_path: str) -> None:
        """
        Plot sentiment trends over time.
        
        Args:
            sentiment_data: DataFrame containing sentiment scores and dates
            save_path: Path to save the plot
        """
        plt.figure(figsize=(14, 8))
        
        # Ensure date column is datetime
        sentiment_data['date'] = pd.to_datetime(sentiment_data['date'])
        
        # Group by date and calculate daily average sentiment
        daily_sentiment = sentiment_data.groupby('date')['compound_score'].agg(['mean', 'count']).reset_index()
        
        # Plot trend line
        sns.lineplot(data=daily_sentiment, x='date', y='mean', alpha=0.8)
        
        # Add rolling average
        daily_sentiment['rolling_mean'] = daily_sentiment['mean'].rolling(window=30, center=True).mean()
        plt.plot(daily_sentiment['date'], daily_sentiment['rolling_mean'], 
                color='red', linewidth=2, label='30-day Moving Average')
        
        plt.title('Sentiment Trends Over Time', fontweight='bold', pad=20)
        plt.xlabel('Date')
        plt.ylabel('Average Sentiment Score')
        plt.legend()
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved sentiment trends plot to {save_path}")
        
    def price_vs_sentiment(self, sentiment_data: pd.DataFrame, save_path: str) -> None:
        """
        Plot relationship between price and sentiment.
        
        Args:
            sentiment_data: DataFrame containing sentiment scores and prices
            save_path: Path to save the plot
        """
        plt.figure(figsize=self.figure_size)
        
        # Remove outliers for better visualization
        q99 = sentiment_data['price'].quantile(0.99)
        filtered_data = sentiment_data[sentiment_data['price'] <= q99]
        
        sns.scatterplot(data=filtered_data, x='price', y='compound_score', alpha=0.6)
        
        # Add trend line
        z = np.polyfit(filtered_data['price'], filtered_data['compound_score'], 1)
        p = np.poly1d(z)
        plt.plot(filtered_data['price'], p(filtered_data['price']), "r--", alpha=0.8, linewidth=2)
        
        # Calculate correlation
        correlation = filtered_data['price'].corr(filtered_data['compound_score'])
        plt.text(0.05, 0.95, f'Correlation: {correlation:.3f}', 
                transform=plt.gca().transAxes, fontsize=12, 
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        plt.title('Price vs Sentiment Score Relationship', fontweight='bold', pad=20)
        plt.xlabel('Price ($)')
        plt.ylabel('Sentiment Score')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved price vs sentiment plot to {save_path}")
        
    def satisfaction_correlations(self, satisfaction_data: pd.DataFrame, save_path: str) -> None:
        """
        Plot correlation heatmap of satisfaction components.
        
        Args:
            satisfaction_data: DataFrame containing satisfaction scores and components
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Calculate correlation matrix
        correlation_matrix = satisfaction_data.corr()
        
        # Create heatmap
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.3f', cbar_kws={"shrink": .8})
        
        plt.title('Satisfaction Components Correlation Matrix', fontweight='bold', pad=20)
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved satisfaction correlations plot to {save_path}")
        
    def forecast_plot(self, historical_data: pd.Series, forecast_data: dict, save_path: str) -> None:
        """
        Plot satisfaction forecast with confidence intervals.
        
        Args:
            historical_data: Historical satisfaction scores
            forecast_data: Dictionary containing forecast results
            save_path: Path to save the plot
        """
        plt.figure(figsize=(14, 8))
        
        # Plot historical data
        historical_dates = pd.date_range(start='2023-01-01', periods=len(historical_data), freq='D')
        plt.plot(historical_dates, historical_data, label='Historical Data', color='blue', alpha=0.8)
        
        # Plot forecast
        forecast_dates = pd.to_datetime(forecast_data['dates'])
        forecast_values = forecast_data['forecast']
        lower_bound = forecast_data['lower_bound']
        upper_bound = forecast_data['upper_bound']
        
        plt.plot(forecast_dates, forecast_values, color='red', label='Forecast', linewidth=2)
        plt.fill_between(forecast_dates, lower_bound, upper_bound, 
                        color='red', alpha=0.2, label='95% Confidence Interval')
        
        plt.title('Satisfaction Score Forecast', fontweight='bold', pad=20)
        plt.xlabel('Date')
        plt.ylabel('Satisfaction Score')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=self.dpi, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved forecast plot to {save_path}") 