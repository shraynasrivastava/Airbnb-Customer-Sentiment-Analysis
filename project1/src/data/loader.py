import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class AirbnbDataLoader:
    def __init__(self, data_dir: str = "data"):
        """
        Initialize the data loader
        Args:
            data_dir: Directory containing the data files
        """
        self.data_dir = Path(data_dir)

    def load_listings(self, sample_size=None):
        """Load and preprocess listings data"""
        try:
            # Load listings
            listings_df = pd.read_csv(self.data_dir / "listings.csv")
            
            if sample_size:
                listings_df = listings_df.sample(
                    n=min(sample_size, len(listings_df)),
                    random_state=42
                )
            
            # Preprocess
            listings_df['reviews_per_month'] = listings_df['reviews_per_month'].fillna(0)
            listings_df['bathrooms'] = listings_df['bathrooms'].fillna(1)
            listings_df['bedrooms'] = listings_df['bedrooms'].fillna(1)
            listings_df['beds'] = listings_df['beds'].fillna(1)
            
            # Convert price to numeric
            listings_df['price'] = listings_df['price'].str.replace('$', '').str.replace(',', '').astype(float)
            
            # Ensure neighborhood column exists
            if 'neighbourhood' in listings_df.columns:
                listings_df['neighborhood'] = listings_df['neighbourhood']
            elif 'neighborhood' in listings_df.columns:
                pass
            else:
                listings_df['neighborhood'] = 'Unknown'
            
            logger.info(f"Loaded {len(listings_df)} listings")
            return listings_df
            
        except Exception as e:
            logger.error(f"Error loading listings: {e}")
            return None

    def load_reviews(self, listing_ids=None):
        """Load and preprocess reviews data"""
        try:
            # Load reviews
            reviews_df = pd.read_csv(self.data_dir / "reviews.csv")
            
            # Filter for specific listings if provided
            if listing_ids:
                reviews_df = reviews_df[reviews_df['listing_id'].isin(listing_ids)]
            
            # Preprocess
            reviews_df['date'] = pd.to_datetime(reviews_df['date'])
            reviews_df = reviews_df.sort_values('date')
            reviews_df = reviews_df.dropna(subset=['comments'])
            
            logger.info(f"Loaded {len(reviews_df)} reviews")
            return reviews_df
            
        except Exception as e:
            logger.error(f"Error loading reviews: {e}")
            return None

    def load_calendar(self, listing_ids=None):
        """Load and preprocess calendar data"""
        try:
            # Load calendar
            calendar_df = pd.read_csv(self.data_dir / "calendar.csv")
            
            # Filter for specific listings if provided
            if listing_ids:
                calendar_df = calendar_df[calendar_df['listing_id'].isin(listing_ids)]
            
            # Preprocess
            calendar_df['date'] = pd.to_datetime(calendar_df['date'])
            calendar_df['price'] = calendar_df['price'].str.replace('$', '').str.replace(',', '').astype(float)
            calendar_df['available'] = calendar_df['available'].map({'t': True, 'f': False})
            
            logger.info(f"Loaded {len(calendar_df)} calendar entries")
            return calendar_df
            
        except Exception as e:
            logger.error(f"Error loading calendar: {e}")
            return None

    def merge_data(self, reviews_df, listings_df):
        """Merge listings and reviews data"""
        try:
            merged_df = pd.merge(
                reviews_df,
                listings_df[['id', 'neighborhood', 'room_type', 'price', 'host_id']],
                left_on='listing_id',
                right_on='id',
                how='left'
            )
            return merged_df
        except Exception as e:
            logger.error(f"Error merging data: {e}")
            return None 