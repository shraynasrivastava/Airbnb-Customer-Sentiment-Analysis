#!/usr/bin/env python3
"""
Airbnb Sentiment Analysis & Satisfaction Forecasting

Main script to run the complete analysis pipeline including:
- Data loading and preprocessing
- Sentiment analysis using VADER (and optionally BERT)
- Satisfaction score calculation
- Time-series forecasting
- Comprehensive visualization and reporting

Author: Aniket Gupta
Date: 2024
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path for imports
sys.path.append(str(Path(__file__).parent / 'src'))

from config import config
from utils.logging_config import setup_logging, get_logger
from data.loader import AirbnbDataLoader
from analysis.sentiment import SentimentAnalyzer
from analysis.satisfaction import SatisfactionCalculator
from forecasting.forecaster import SatisfactionForecaster
from visualization.plots import PlotGenerator

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Airbnb Sentiment Analysis & Satisfaction Forecasting')
    parser.add_argument('--demo', action='store_true', 
                       help='Run in demo mode with 20 sample listings')
    parser.add_argument('--sample-size', type=int, default=None,
                       help='Number of listings to analyze (default: all)')
    parser.add_argument('--use-bert', action='store_true',
                       help='Use BERT for sentiment analysis (slower but more accurate)')
    parser.add_argument('--config', type=str, default='config.yaml',
                       help='Path to configuration file')
    parser.add_argument('--log-level', type=str, default='INFO',
                       choices=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
                       help='Logging level')
    return parser.parse_args()

def create_output_directories():
    """Create all necessary output directories."""
    directories = [
        config.get('output.plots_dir'),
        config.get('output.reports_dir'),
        config.get('output.models_dir'),
        config.get('output.logs_dir')
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)

def generate_comprehensive_report(sentiment_data, satisfaction_scores, forecast_data, args):
    """Generate a comprehensive analysis report."""
    logger = get_logger(__name__)
    
    try:
        # Calculate comprehensive statistics
        report = {
            'metadata': {
                'analysis_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'version': '1.0.0',
                'configuration': {
                    'demo_mode': args.demo,
                    'sample_size': args.sample_size,
                    'use_bert': args.use_bert,
                    'config_file': args.config
                }
            },
            'data_summary': {
                'total_listings': len(sentiment_data['listing_id'].unique()),
                'total_reviews': len(sentiment_data),
                'date_range': {
                    'start': sentiment_data['date'].min().strftime('%Y-%m-%d'),
                    'end': sentiment_data['date'].max().strftime('%Y-%m-%d')
                },
                'average_reviews_per_listing': len(sentiment_data) / len(sentiment_data['listing_id'].unique())
            },
            'sentiment_analysis': {
                'overall_sentiment': {
                    'mean': float(sentiment_data['compound_score'].mean()),
                    'median': float(sentiment_data['compound_score'].median()),
                    'std': float(sentiment_data['compound_score'].std())
                },
                'sentiment_distribution': sentiment_data['sentiment'].value_counts().to_dict(),
                'top_positive_neighborhoods': sentiment_data.groupby('neighborhood')['compound_score'].mean().nlargest(5).to_dict(),
                'bottom_neighborhoods': sentiment_data.groupby('neighborhood')['compound_score'].mean().nsmallest(3).to_dict()
            },
            'satisfaction_analysis': {
                'overall_satisfaction': {
                    'mean': float(satisfaction_scores['satisfaction_score'].mean()) if 'satisfaction_score' in satisfaction_scores.columns else None,
                    'top_10_percent_threshold': float(satisfaction_scores['satisfaction_score'].quantile(0.9)) if 'satisfaction_score' in satisfaction_scores.columns else None
                }
            },
            'forecast_results': forecast_data if forecast_data else {},
            'business_insights': {
                'price_sentiment_correlation': float(sentiment_data['price'].corr(sentiment_data['compound_score'])),
                'neighborhood_performance': sentiment_data.groupby('neighborhood').agg({
                    'compound_score': 'mean',
                    'price': 'mean',
                    'listing_id': 'count'
                }).round(3).to_dict(),
                'recommendations': generate_business_recommendations(sentiment_data, satisfaction_scores)
            }
        }
        
        # Save comprehensive report
        report_path = Path(config.get('output.reports_dir')) / 'comprehensive_analysis_report.json'
        with open(report_path, 'w') as f:
            json.dump(report, f, indent=4, default=str)
        
        logger.info(f"Comprehensive report saved to {report_path}")
        return report
        
    except Exception as e:
        logger.error(f"Error generating comprehensive report: {str(e)}")
        return None

def generate_business_recommendations(sentiment_data, satisfaction_scores):
    """Generate actionable business recommendations."""
    recommendations = []
    
    # Price vs sentiment insights
    price_sentiment_corr = sentiment_data['price'].corr(sentiment_data['compound_score'])
    if price_sentiment_corr < -0.1:
        recommendations.append("Consider pricing strategies as higher prices correlate with lower sentiment")
    
    # Neighborhood insights
    neighborhood_performance = sentiment_data.groupby('neighborhood')['compound_score'].mean()
    best_neighborhood = neighborhood_performance.idxmax()
    worst_neighborhood = neighborhood_performance.idxmin()
    
    recommendations.extend([
        f"Focus marketing efforts on {best_neighborhood} which shows highest sentiment scores",
        f"Investigate service quality issues in {worst_neighborhood} neighborhood",
        "Implement sentiment monitoring dashboard for real-time feedback tracking"
    ])
    
    return recommendations

def print_analysis_summary(report):
    """Print a summary of the analysis results."""
    logger = get_logger(__name__)
    
    print("\n" + "="*60)
    print("           AIRBNB SENTIMENT ANALYSIS SUMMARY")
    print("="*60)
    
    if report:
        data_summary = report['data_summary']
        sentiment_summary = report['sentiment_analysis']
        
        print(f"📊 Dataset Overview:")
        print(f"   • Total Listings: {data_summary['total_listings']:,}")
        print(f"   • Total Reviews: {data_summary['total_reviews']:,}")
        print(f"   • Date Range: {data_summary['date_range']['start']} to {data_summary['date_range']['end']}")
        print(f"   • Avg Reviews per Listing: {data_summary['average_reviews_per_listing']:.1f}")
        
        print(f"\n🎯 Sentiment Analysis:")
        overall = sentiment_summary['overall_sentiment']
        print(f"   • Average Sentiment: {overall['mean']:.3f} (Range: -1 to +1)")
        print(f"   • Sentiment Std Dev: {overall['std']:.3f}")
        
        distribution = sentiment_summary['sentiment_distribution']
        total_reviews = sum(distribution.values())
        for sentiment, count in distribution.items():
            percentage = (count / total_reviews) * 100
            print(f"   • {sentiment.title()}: {count:,} ({percentage:.1f}%)")
        
        print(f"\n🏆 Top Performing Neighborhoods:")
        for i, (neighborhood, score) in enumerate(list(sentiment_summary['top_positive_neighborhoods'].items())[:3], 1):
            print(f"   {i}. {neighborhood}: {score:.3f}")
        
        if 'business_insights' in report:
            insights = report['business_insights']
            print(f"\n💡 Key Insights:")
            print(f"   • Price-Sentiment Correlation: {insights['price_sentiment_correlation']:.3f}")
            
            print(f"\n📋 Recommendations:")
            for i, rec in enumerate(insights['recommendations'][:3], 1):
                print(f"   {i}. {rec}")
    
    print("\n" + "="*60)
    print("Analysis complete! Check the outputs/ directory for detailed results.")
    print("="*60 + "\n")

def main():
    """Main execution function."""
    args = parse_arguments()
    
    # Update configuration with command line arguments
    if args.demo:
        config.set('data.demo_mode', True)
        config.set('data.sample_size', 20)
    elif args.sample_size:
        config.set('data.sample_size', args.sample_size)
    
    if args.use_bert:
        config.set('analysis.sentiment.use_transformers', True)
    
    # Set up logging
    setup_logging(log_level=args.log_level, log_dir=config.get('output.logs_dir'))
    logger = get_logger(__name__)
    
    logger.info("Starting Airbnb Sentiment Analysis & Satisfaction Forecasting")
    logger.info(f"Configuration: Demo={args.demo}, Sample={args.sample_size}, BERT={args.use_bert}")
    
    try:
        # Create output directories
        create_output_directories()
        
        # Initialize components
        logger.info("Initializing analysis components...")
        data_loader = AirbnbDataLoader(data_dir=config.get('data.raw_data_dir'))
        sentiment_analyzer = SentimentAnalyzer(use_transformers=config.get('analysis.sentiment.use_transformers'))
        satisfaction_calc = SatisfactionCalculator()
        forecaster = SatisfactionForecaster()
        plot_generator = PlotGenerator(
            style=config.get('visualization.style'),
            figure_size=config.get('visualization.figure_size'),
            font_size=config.get('visualization.font_size'),
            dpi=config.get('visualization.dpi')
        )
        
        # Load and process data
        logger.info("Loading data...")
        sample_size = config.get('data.sample_size')
        listings_df = data_loader.load_listings(sample_size)
        reviews_df = data_loader.load_reviews(listings_df['id'].tolist())
        calendar_df = data_loader.load_calendar(listings_df['id'].tolist())
        
        # Analyze sentiment
        logger.info("Analyzing sentiment...")
        sentiment_data = sentiment_analyzer.analyze_reviews(reviews_df)
        sentiment_data = data_loader.merge_data(sentiment_data, listings_df)
        
        # Generate visualizations
        logger.info("Generating visualizations...")
        plots_dir = Path(config.get('output.plots_dir'))
        
        plot_generator.sentiment_distribution(sentiment_data, plots_dir / 'sentiment_distribution.png')
        plot_generator.sentiment_by_neighborhood(sentiment_data, plots_dir / 'sentiment_by_neighborhood.png')
        plot_generator.sentiment_trends(sentiment_data, plots_dir / 'sentiment_trends.png')
        plot_generator.price_vs_sentiment(sentiment_data, plots_dir / 'price_vs_sentiment.png')
        
        # Calculate satisfaction scores
        logger.info("Calculating satisfaction scores...")
        satisfaction_scores = satisfaction_calc.calculate_satisfaction(
            sentiment_data, listings_df, calendar_df
        )
        
        if satisfaction_scores is not None:
            plot_generator.satisfaction_correlations(satisfaction_scores, plots_dir / 'satisfaction_correlations.png')
        
        # Generate forecast
        logger.info("Generating satisfaction forecast...")
        forecast_data = None
        if satisfaction_scores is not None and 'satisfaction_score' in satisfaction_scores.columns:
            forecast_data = forecaster.forecast_satisfaction(
                satisfaction_scores['satisfaction_score'],
                periods=config.get('forecasting.periods')
            )
            
            if forecast_data:
                plot_generator.forecast_plot(
                    satisfaction_scores['satisfaction_score'],
                    forecast_data,
                    plots_dir / 'satisfaction_forecast.png'
                )
        
        # Generate comprehensive report
        logger.info("Generating comprehensive report...")
        report = generate_comprehensive_report(sentiment_data, satisfaction_scores, forecast_data, args)
        
        # Print summary
        print_analysis_summary(report)
        
        logger.info("Analysis pipeline completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in main execution: {str(e)}")
        raise

if __name__ == "__main__":
    main() 