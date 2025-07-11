import pandas as pd
import numpy as np
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
import re
from typing import Dict, Any, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SentimentAnalyzer:
    def __init__(self, use_transformers: bool = False):
        """Initialize sentiment analyzer with option to use BERT"""
        self.vader = SentimentIntensityAnalyzer()
        self.use_transformers = use_transformers
        if use_transformers:
            try:
                self.bert_analyzer = pipeline("sentiment-analysis")
            except Exception as e:
                logger.error(f"Failed to initialize BERT analyzer: {str(e)}")
                self.use_transformers = False
        
        # Initialize NLTK resources with proper error handling
        self._initialize_nltk_resources()

    def _initialize_nltk_resources(self) -> None:
        """Initialize NLTK resources with proper error handling"""
        required_resources = {
            'punkt': 'tokenizers/punkt',
            'stopwords': 'corpora/stopwords'
        }
        
        for resource, path in required_resources.items():
            try:
                nltk.data.find(path)
            except LookupError:
                logger.info(f"Downloading {resource}...")
                try:
                    nltk.download(resource, quiet=True)
                except Exception as e:
                    logger.error(f"Failed to download {resource}: {str(e)}")
                    raise RuntimeError(f"Failed to initialize required NLTK resource: {resource}")
        
        try:
            self.stop_words = set(stopwords.words('english'))
        except Exception as e:
            logger.error(f"Failed to load stopwords: {str(e)}")
            self.stop_words = set()

    def preprocess_text(self, text: str) -> str:
        """Clean and preprocess text data with robust error handling"""
        if pd.isna(text):
            return ""
        
        try:
            # Convert to lowercase and string type
            text = str(text).lower()
            
            # Remove special characters and numbers
            text = re.sub(r'[^a-zA-Z\s]', '', text)
            
            try:
                # Split into sentences and tokenize
                words = []
                for sentence in sent_tokenize(text):
                    try:
                        words.extend(word_tokenize(sentence))
                    except Exception as e:
                        logger.warning(f"Error tokenizing sentence: {str(e)}")
                        # Fallback to simple space-based tokenization
                        words.extend(sentence.split())
                
                # Remove stopwords
                words = [word for word in words if word not in self.stop_words]
                
                return ' '.join(words)
            
            except Exception as e:
                logger.warning(f"Error during tokenization: {str(e)}")
                # Fallback to simple preprocessing
                return ' '.join([word for word in text.split() if word not in self.stop_words])
                
        except Exception as e:
            logger.error(f"Error preprocessing text: {str(e)}")
            return text

    def get_vader_sentiment(self, text: str) -> Dict[str, float]:
        """Get sentiment scores using VADER with error handling"""
        try:
            return self.vader.polarity_scores(text)
        except Exception as e:
            logger.error(f"Error getting VADER sentiment: {str(e)}")
            return {'compound': 0.0, 'neg': 0.0, 'neu': 1.0, 'pos': 0.0}

    def get_bert_sentiment(self, text: str) -> Dict[str, Any]:
        """Get sentiment using BERT with error handling"""
        if not self.use_transformers:
            raise ValueError("BERT analyzer not initialized. Set use_transformers=True")
        
        try:
            # Truncate text to BERT's maximum token limit (512)
            result = self.bert_analyzer(text[:512])[0]
            return {
                'label': result['label'],
                'score': result['score']
            }
        except Exception as e:
            logger.error(f"Error getting BERT sentiment: {str(e)}")
            return {
                'label': 'NEUTRAL',
                'score': 0.5
            }

    def analyze_reviews(self, reviews_df: pd.DataFrame, text_column: str = 'comments') -> pd.DataFrame:
        """Analyze sentiment for all reviews with progress tracking"""
        if text_column not in reviews_df.columns:
            raise ValueError(f"Text column '{text_column}' not found in DataFrame")
            
        total_reviews = len(reviews_df)
        logger.info(f"Starting sentiment analysis for {total_reviews} reviews...")
        
        # Preprocess reviews
        logger.info("Preprocessing reviews...")
        reviews_df['processed_text'] = reviews_df[text_column].apply(self.preprocess_text)
        
        # Get VADER sentiment
        logger.info("Analyzing sentiment with VADER...")
        vader_scores = reviews_df['processed_text'].apply(self.get_vader_sentiment)
        reviews_df['compound_score'] = vader_scores.apply(lambda x: x['compound'])
        reviews_df['sentiment'] = reviews_df['compound_score'].apply(
            lambda x: 'positive' if x > 0.05 else ('negative' if x < -0.05 else 'neutral')
        )
        
        # Get BERT sentiment if enabled
        if self.use_transformers:
            logger.info("Analyzing sentiment with BERT...")
            bert_results = reviews_df['processed_text'].apply(self.get_bert_sentiment)
            reviews_df['bert_label'] = bert_results.apply(lambda x: x['label'])
            reviews_df['bert_score'] = bert_results.apply(lambda x: x['score'])
        
        logger.info("Sentiment analysis completed!")
        return reviews_df

    def get_sentiment_summary(self, reviews_df: pd.DataFrame) -> Dict[str, Any]:
        """Generate summary statistics of sentiment analysis"""
        try:
            summary = {
                'total_reviews': len(reviews_df),
                'sentiment_dist': reviews_df['sentiment'].value_counts().to_dict(),
                'avg_compound_score': reviews_df['compound_score'].mean()
            }
            
            if self.use_transformers and 'bert_label' in reviews_df.columns:
                summary['bert_sentiment_dist'] = reviews_df['bert_label'].value_counts().to_dict()
                summary['bert_avg_score'] = reviews_df['bert_score'].mean()
                
            return summary
        except Exception as e:
            logger.error(f"Error generating sentiment summary: {str(e)}")
            return {
                'total_reviews': 0,
                'sentiment_dist': {},
                'avg_compound_score': 0.0
            } 