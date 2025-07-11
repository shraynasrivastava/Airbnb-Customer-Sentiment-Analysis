# Methodology

## Overview

This document outlines the methodology used in the Airbnb Sentiment Analysis & Satisfaction Forecasting project, detailing the techniques, algorithms, and approaches employed.

## 1. Data Collection and Preprocessing

### Data Sources
- **Seattle Airbnb Open Data**: Available from Kaggle
- **Components**: listings.csv, reviews.csv, calendar.csv
- **Time Period**: Historical data spanning multiple years

### Data Preprocessing Steps

1. **Text Cleaning**
   - Convert to lowercase
   - Remove special characters and numbers
   - Remove stopwords using NLTK
   - Tokenization using NLTK's punkt tokenizer

2. **Data Validation**
   - Handle missing values appropriately
   - Convert price strings to numeric values
   - Standardize date formats
   - Filter out reviews without text content

3. **Feature Engineering**
   - Merge datasets on listing_id
   - Create time-based features
   - Calculate derived metrics (reviews per month, etc.)

## 2. Sentiment Analysis

### Primary Method: VADER (Valence Aware Dictionary and sEntiment Reasoner)

**Why VADER?**
- Specifically designed for social media text
- Handles emoticons, slang, and punctuation
- Provides compound scores ranging from -1 (most negative) to +1 (most positive)
- No training required, making it efficient for production use

**VADER Components:**
- Negative score (neg)
- Neutral score (neu) 
- Positive score (pos)
- Compound score (primary metric used)

### Optional Enhancement: BERT

**BERT Implementation:**
- Uses pre-trained transformers from Hugging Face
- Provides more nuanced understanding of context
- Computationally expensive but more accurate
- Available as an option via `--use-bert` flag

### Sentiment Classification
```
if compound_score > 0.05:
    sentiment = 'positive'
elif compound_score < -0.05:
    sentiment = 'negative'
else:
    sentiment = 'neutral'
```

## 3. Satisfaction Score Calculation

### Weighted Satisfaction Index

The satisfaction score combines multiple factors using a weighted approach:

```
Satisfaction Score = 
    0.5 × Normalized_Sentiment_Score +
    0.2 × Normalized_Review_Count +
    0.2 × Normalized_Price_Inverse +
    0.1 × Normalized_Availability
```

### Component Explanations

1. **Sentiment Score (50% weight)**
   - Primary driver of satisfaction
   - Based on average compound sentiment of all reviews

2. **Review Count (20% weight)**
   - More reviews indicate higher activity and trustworthiness
   - Normalized using MinMaxScaler

3. **Price Competitiveness (20% weight)**
   - Inverse relationship: lower prices = better satisfaction
   - Calculated as 1/(price + 1) to avoid division by zero

4. **Availability (10% weight)**
   - Higher availability indicates better service
   - Based on calendar data or listing availability

### Normalization
All components are normalized to [0,1] range using MinMaxScaler to ensure equal contribution according to weights.

## 4. Time Series Forecasting

### Polynomial Regression Approach

**Algorithm:** Scikit-learn's LinearRegression with PolynomialFeatures

**Why Polynomial Regression?**
- Simple and interpretable
- Handles non-linear trends
- Computationally efficient
- Provides confidence intervals

**Implementation Details:**
- Degree 2 polynomial features
- 180-day forecast horizon (configurable)
- 95% confidence intervals
- Mean Squared Error (MSE) for model evaluation

**Forecast Components:**
- Trend: Overall direction of satisfaction scores
- Confidence Intervals: 95% prediction bounds
- Performance Metrics: MSE and standard error

### Alternative Approach Note
While the documentation mentions Prophet, the current implementation uses polynomial regression for simplicity and efficiency. Prophet could be integrated as an advanced option for seasonal decomposition.

## 5. Visualization Strategy

### Chart Types and Purpose

1. **Distribution Plots**
   - Histogram with KDE for sentiment distribution
   - Shows overall sentiment patterns

2. **Box Plots**
   - Sentiment by neighborhood comparison
   - Identifies regional performance differences

3. **Time Series Plots**
   - Sentiment trends over time
   - Includes moving averages for trend identification

4. **Scatter Plots**
   - Price vs sentiment relationships
   - Includes correlation analysis and trend lines

5. **Heatmaps**
   - Correlation matrices for satisfaction components
   - Feature importance visualization

6. **Forecast Plots**
   - Historical data with future predictions
   - Confidence interval visualization

## 6. Statistical Analysis

### Correlation Analysis
- Pearson correlation between price and sentiment
- Component correlation matrix for satisfaction scores
- Neighborhood performance comparisons

### Business Insights Generation
- Top/bottom performing neighborhoods
- Price sensitivity analysis
- Recommendation algorithms based on statistical findings

## 7. Model Validation

### Performance Metrics
- **Sentiment Analysis**: Manual validation against sample reviews
- **Forecasting**: MSE, standard error, confidence interval coverage
- **Satisfaction Scores**: Component contribution analysis

### Cross-Validation
- Time-based validation for forecasting models
- Neighborhood-based validation for sentiment analysis

## 8. Limitations and Considerations

### Known Limitations
1. **Language**: Only English reviews supported
2. **Bias**: VADER may not capture cultural nuances
3. **Temporal**: Historical patterns may not predict future changes
4. **Sample Size**: Results depend on sufficient review volume

### Mitigation Strategies
1. **BERT Option**: More sophisticated NLP when computational resources allow
2. **Configurable Weights**: Allows adjustment of satisfaction components
3. **Confidence Intervals**: Quantifies uncertainty in forecasts
4. **Comprehensive Reporting**: Transparent methodology documentation

## 9. Future Enhancements

### Potential Improvements
1. **Advanced Forecasting**: Integration with Facebook Prophet
2. **Multi-language Support**: Extend sentiment analysis to other languages
3. **Deep Learning**: Custom LSTM models for time series
4. **Real-time Processing**: Streaming analytics capabilities
5. **Advanced Features**: Topic modeling, aspect-based sentiment analysis

This methodology provides a robust foundation for sentiment analysis and satisfaction forecasting while maintaining transparency and reproducibility. 