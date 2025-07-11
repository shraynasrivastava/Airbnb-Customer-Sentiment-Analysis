import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import logging

logger = logging.getLogger(__name__)

class SatisfactionCalculator:
    def __init__(self):
        """Initialize with default weights for different factors"""
        self.weights = {
            'sentiment_score': 0.5,
            'review_count': 0.2,
            'price_inverse': 0.2,
            'availability': 0.1
        }
        self.scaler = MinMaxScaler()
        
    def normalize_feature(self, series):
        """Normalize a feature to 0-1 range"""
        return pd.Series(self.scaler.fit_transform(series.values.reshape(-1, 1)).flatten(), index=series.index)
    
    def calculate_satisfaction(self, sentiment_df, listings_df, calendar_df):
        """Calculate satisfaction scores for listings"""
        try:
            # 1. Average sentiment per listing
            avg_sentiment = sentiment_df.groupby('listing_id')['compound_score'].mean()
            sentiment_score = self.normalize_feature(avg_sentiment)
            
            # 2. Review counts
            review_counts = sentiment_df.groupby('listing_id').size()
            review_score = self.normalize_feature(review_counts)
            
            # 3. Price competitiveness (inverse - lower price is better)
            price_inverse = 1 / (listings_df['price'] + 1)  # Add 1 to avoid division by zero
            price_score = self.normalize_feature(price_inverse)
            
            # 4. Availability score from calendar
            availability = calendar_df.groupby('listing_id')['available'].mean()
            availability_score = availability.fillna(0.5)  # Default to 0.5 if no calendar data
            
            # Calculate weighted satisfaction score
            satisfaction_scores = pd.DataFrame({
                'listing_id': listings_df['id'],
                'sentiment_score': sentiment_score,
                'review_score': review_score,
                'price_score': price_score,
                'availability_score': availability_score,
                'satisfaction_score': (
                    self.weights['sentiment_score'] * sentiment_score +
                    self.weights['review_count'] * review_score +
                    self.weights['price_inverse'] * price_score +
                    self.weights['availability'] * availability_score
                )
            }).set_index('listing_id')
            
            logger.info(f"Calculated satisfaction scores for {len(satisfaction_scores)} listings")
            return satisfaction_scores
            
        except Exception as e:
            logger.error(f"Error calculating satisfaction scores: {e}")
            return None
    
    def calculate_satisfaction_index(self, listings_df, sentiment_df):
        """Calculate satisfaction index for each listing"""
        # Prepare features
        features = {}
        
        # 1. Sentiment Score
        avg_sentiment = sentiment_df.groupby('listing_id')['compound_score'].mean()
        features['sentiment_score'] = self.normalize_feature(avg_sentiment)
        
        # 2. Review Count
        review_counts = sentiment_df.groupby('listing_id').size()
        features['review_count'] = self.normalize_feature(review_counts)
        
        # 3. Price (inverse - lower price is better)
        price_inverse = 1 / (listings_df['price'] + 1)  # Add 1 to avoid division by zero
        features['price_inverse'] = self.normalize_feature(price_inverse)
        
        # 4. Availability (from listings)
        if 'availability_365' in listings_df.columns:
            availability = listings_df['availability_365'] / 365
            features['availability'] = availability
        else:
            features['availability'] = pd.Series(1.0, index=listings_df.index)
        
        # Calculate weighted satisfaction index
        satisfaction_index = pd.Series(0.0, index=listings_df.index)
        for feature, weight in self.weights.items():
            if feature in features:
                satisfaction_index += weight * features[feature]
        
        return satisfaction_index
    
    def get_satisfaction_insights(self, listings_df, sentiment_df, satisfaction_scores):
        """Generate insights based on satisfaction scores"""
        try:
            insights = {
                'avg_satisfaction': float(satisfaction_scores['satisfaction_score'].mean()),
                'top_neighborhoods': listings_df.groupby('neighborhood')['satisfaction_score'].mean().nlargest(5).to_dict(),
                'top_room_types': listings_df.groupby('room_type')['satisfaction_score'].mean().nlargest(3).to_dict(),
                'sentiment_correlation': float(
                    satisfaction_scores['satisfaction_score'].corr(sentiment_df.groupby('listing_id')['compound_score'].mean())
                )
            }
            return insights
        except Exception as e:
            logger.error(f"Error generating satisfaction insights: {e}")
            return None
    
    def get_improvement_recommendations(self, listing_id, listings_df, sentiment_df, satisfaction_scores):
        """Generate recommendations for improving satisfaction"""
        try:
            listing = listings_df.loc[listing_id]
            listing_sentiment = sentiment_df[sentiment_df['listing_id'] == listing_id]
            listing_satisfaction = satisfaction_scores.loc[listing_id]
            
            recommendations = []
            
            # Check sentiment scores
            if listing_sentiment['compound_score'].mean() < sentiment_df['compound_score'].mean():
                recommendations.append("Improve guest experience based on review feedback")
                
            # Check pricing
            neighborhood_avg_price = listings_df[listings_df['neighborhood'] == listing['neighborhood']]['price'].mean()
            if listing['price'] > neighborhood_avg_price * 1.2:  # 20% above neighborhood average
                recommendations.append("Consider adjusting price to be more competitive with neighborhood average")
                
            # Check review count
            if len(listing_sentiment) < sentiment_df.groupby('listing_id').size().mean():
                recommendations.append("Encourage more guest reviews to build trust")
                
            # Check availability
            if listing.get('availability_365', 0) < 180:  # Less than 6 months availability
                recommendations.append("Consider increasing availability to attract more bookings")
                
            return recommendations
        except Exception as e:
            logger.error(f"Error generating recommendations: {e}")
            return [] 