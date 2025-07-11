import pandas as pd
from sentiment_analyzer import SentimentAnalyzer
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_sentiment_analyzer():
    """Test the sentiment analyzer with a small sample of reviews"""
    # Create sample reviews
    sample_reviews = pd.DataFrame({
        'comments': [
            "This place was absolutely amazing! Great location and wonderful host.",
            "Terrible experience. The room was dirty and the host was unresponsive.",
            "It was okay, nothing special but got the job done.",
            "Really enjoyed our stay! The apartment was clean and comfortable.",
            "Mixed feelings about this one. Good location but noisy neighbors."
        ]
    })

    try:
        # Initialize analyzer with both VADER and BERT
        logger.info("Initializing sentiment analyzer...")
        analyzer = SentimentAnalyzer(use_transformers=True)
        
        # Analyze reviews
        logger.info("Analyzing sample reviews...")
        results_df = analyzer.analyze_reviews(sample_reviews)
        
        # Get summary
        summary = analyzer.get_sentiment_summary(results_df)
        
        # Print results
        logger.info("\nAnalysis Results:")
        logger.info(f"Total reviews analyzed: {summary['total_reviews']}")
        logger.info("\nVADER Sentiment Distribution:")
        for sentiment, count in summary['vader_sentiment_dist'].items():
            logger.info(f"{sentiment}: {count}")
        
        if 'bert_sentiment_dist' in summary:
            logger.info("\nBERT Sentiment Distribution:")
            for sentiment, count in summary['bert_sentiment_dist'].items():
                logger.info(f"{sentiment}: {count}")
        
        logger.info("\nDetailed Results:")
        for idx, row in results_df.iterrows():
            logger.info(f"\nReview {idx + 1}:")
            logger.info(f"Text: {row['comments']}")
            logger.info(f"VADER Sentiment: {row['vader_sentiment']} (compound: {row['vader_compound']:.2f})")
            if 'bert_label' in row:
                logger.info(f"BERT Sentiment: {row['bert_label']} (score: {row['bert_score']:.2f})")
        
        return True
        
    except Exception as e:
        logger.error(f"Error during sentiment analysis test: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_sentiment_analyzer()
    if success:
        logger.info("\nSentiment analyzer test completed successfully!")
    else:
        logger.error("\nSentiment analyzer test failed. Please check the errors above.") 