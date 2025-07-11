# API Reference

This document provides comprehensive API documentation for all modules in the Airbnb Sentiment Analysis project.

## Core Modules

### `src.data.loader`

#### `AirbnbDataLoader`

Main class for loading and preprocessing Airbnb dataset files.

```python
class AirbnbDataLoader:
    def __init__(self, data_dir: str = "data")
```

**Parameters:**
- `data_dir` (str): Directory containing the CSV data files

**Methods:**

##### `load_listings(sample_size=None)`
Loads and preprocesses listings data.

**Parameters:**
- `sample_size` (int, optional): Number of listings to sample. If None, loads all.

**Returns:**
- `pandas.DataFrame`: Preprocessed listings data

**Example:**
```python
loader = AirbnbDataLoader("data/raw")
listings = loader.load_listings(sample_size=100)
```

##### `load_reviews(listing_ids=None)`
Loads and preprocesses reviews data.

**Parameters:**
- `listing_ids` (list, optional): List of listing IDs to filter by

**Returns:**
- `pandas.DataFrame`: Preprocessed reviews data

##### `load_calendar(listing_ids=None)`
Loads and preprocesses calendar data.

**Parameters:**
- `listing_ids` (list, optional): List of listing IDs to filter by

**Returns:**
- `pandas.DataFrame`: Preprocessed calendar data

##### `merge_data(reviews_df, listings_df)`
Merges reviews and listings dataframes.

**Parameters:**
- `reviews_df` (pandas.DataFrame): Reviews data
- `listings_df` (pandas.DataFrame): Listings data

**Returns:**
- `pandas.DataFrame`: Merged dataset

---

### `src.analysis.sentiment`

#### `SentimentAnalyzer`

Sentiment analysis using VADER and optional BERT models.

```python
class SentimentAnalyzer:
    def __init__(self, use_transformers: bool = False)
```

**Parameters:**
- `use_transformers` (bool): Whether to use BERT transformer model

**Methods:**

##### `analyze_reviews(reviews_df, text_column='comments')`
Analyzes sentiment for all reviews in a DataFrame.

**Parameters:**
- `reviews_df` (pandas.DataFrame): DataFrame containing review text
- `text_column` (str): Column name containing review text

**Returns:**
- `pandas.DataFrame`: Original data with sentiment scores added

**Added Columns:**
- `processed_text`: Cleaned review text
- `compound_score`: VADER compound sentiment score (-1 to +1)
- `sentiment`: Categorical sentiment (positive/negative/neutral)
- `bert_label`: BERT sentiment label (if transformers enabled)
- `bert_score`: BERT confidence score (if transformers enabled)

**Example:**
```python
analyzer = SentimentAnalyzer(use_transformers=True)
sentiment_data = analyzer.analyze_reviews(reviews_df)
```

##### `preprocess_text(text)`
Cleans and preprocesses text data.

**Parameters:**
- `text` (str): Raw text to preprocess

**Returns:**
- `str`: Cleaned text

##### `get_sentiment_summary(reviews_df)`
Generates summary statistics of sentiment analysis.

**Parameters:**
- `reviews_df` (pandas.DataFrame): DataFrame with sentiment scores

**Returns:**
- `dict`: Summary statistics

---

### `src.analysis.satisfaction`

#### `SatisfactionCalculator`

Multi-factor satisfaction score calculation.

```python
class SatisfactionCalculator:
    def __init__(self)
```

**Attributes:**
- `weights` (dict): Weighting factors for satisfaction components

**Methods:**

##### `calculate_satisfaction(sentiment_df, listings_df, calendar_df)`
Calculates satisfaction scores for listings.

**Parameters:**
- `sentiment_df` (pandas.DataFrame): Sentiment analysis results
- `listings_df` (pandas.DataFrame): Listings data
- `calendar_df` (pandas.DataFrame): Calendar data

**Returns:**
- `pandas.DataFrame`: Satisfaction scores with components

**Columns:**
- `sentiment_score`: Normalized sentiment component
- `review_score`: Normalized review count component
- `price_score`: Normalized price competitiveness component
- `availability_score`: Normalized availability component
- `satisfaction_score`: Weighted final score

**Example:**
```python
calculator = SatisfactionCalculator()
satisfaction_scores = calculator.calculate_satisfaction(
    sentiment_data, listings_df, calendar_df
)
```

##### `get_improvement_recommendations(listing_id, listings_df, sentiment_df, satisfaction_scores)`
Generates improvement recommendations for a specific listing.

**Parameters:**
- `listing_id` (int): ID of the listing
- `listings_df` (pandas.DataFrame): Listings data
- `sentiment_df` (pandas.DataFrame): Sentiment data
- `satisfaction_scores` (pandas.DataFrame): Satisfaction scores

**Returns:**
- `list`: List of recommendation strings

---

### `src.forecasting.forecaster`

#### `SatisfactionForecaster`

Time series forecasting for satisfaction trends.

```python
class SatisfactionForecaster:
    def __init__(self)
```

**Methods:**

##### `forecast_satisfaction(satisfaction_scores, periods=180)`
Generates satisfaction forecast using polynomial regression.

**Parameters:**
- `satisfaction_scores` (pandas.Series): Historical satisfaction scores
- `periods` (int): Number of future periods to forecast

**Returns:**
- `dict`: Forecast results with confidence intervals

**Dictionary Keys:**
- `dates`: List of forecast dates
- `forecast`: Predicted values
- `lower_bound`: Lower confidence bound
- `upper_bound`: Upper confidence bound
- `mse`: Mean squared error
- `std_err`: Standard error

**Example:**
```python
forecaster = SatisfactionForecaster()
forecast = forecaster.forecast_satisfaction(
    satisfaction_scores['satisfaction_score'], 
    periods=90
)
```

---

### `src.visualization.plots`

#### `PlotGenerator`

Professional visualization generation.

```python
class PlotGenerator:
    def __init__(self, style="whitegrid", figure_size=(10, 6), font_size=12, dpi=300)
```

**Parameters:**
- `style` (str): Seaborn style theme
- `figure_size` (tuple): Default figure size
- `font_size` (int): Default font size
- `dpi` (int): Resolution for saved plots

**Methods:**

##### `sentiment_distribution(sentiment_data, save_path)`
Creates sentiment distribution histogram.

**Parameters:**
- `sentiment_data` (pandas.DataFrame): Data with compound_score column
- `save_path` (str): Path to save the plot

##### `sentiment_by_neighborhood(sentiment_data, save_path)`
Creates neighborhood sentiment comparison boxplot.

##### `sentiment_trends(sentiment_data, save_path)`
Creates time series plot of sentiment trends.

##### `price_vs_sentiment(sentiment_data, save_path)`
Creates scatter plot of price vs sentiment relationship.

##### `satisfaction_correlations(satisfaction_data, save_path)`
Creates correlation heatmap of satisfaction components.

##### `forecast_plot(historical_data, forecast_data, save_path)`
Creates forecast visualization with confidence intervals.

**Example:**
```python
plot_generator = PlotGenerator(dpi=300)
plot_generator.sentiment_distribution(
    sentiment_data, 
    'outputs/visualizations/sentiment_dist.png'
)
```

---

### `src.utils.logging_config`

#### Functions

##### `setup_logging(log_level='INFO', log_dir='outputs/logs')`
Configures application logging.

**Parameters:**
- `log_level` (str): Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
- `log_dir` (str): Directory for log files

**Returns:**
- `logging.Logger`: Configured logger instance

##### `get_logger(name)`
Gets a logger instance with the specified name.

**Parameters:**
- `name` (str): Logger name

**Returns:**
- `logging.Logger`: Logger instance

---

### `src.config`

#### `Config` Class

Configuration management using YAML files.

```python
class Config:
    def __init__(self, config_path: str = "config.yaml")
```

**Methods:**

##### `get(key, default=None)`
Gets configuration value using dot notation.

**Parameters:**
- `key` (str): Configuration key (e.g., 'data.sample_size')
- `default`: Default value if key not found

**Returns:**
- `Any`: Configuration value

##### `set(key, value)`
Sets configuration value using dot notation.

**Parameters:**
- `key` (str): Configuration key
- `value` (Any): Value to set

**Example:**
```python
from src.config import config

# Get configuration values
sample_size = config.get('data.sample_size')
use_bert = config.get('analysis.sentiment.use_transformers')

# Set configuration values
config.set('data.sample_size', 1000)
config.set('analysis.sentiment.use_transformers', True)
```

---

## Main Application

### Command Line Interface

The main application provides a comprehensive CLI for running analysis:

```bash
python main.py [OPTIONS]
```

**Options:**
- `--demo`: Run in demo mode with 20 sample listings
- `--sample-size INTEGER`: Number of listings to analyze
- `--use-bert`: Use BERT for sentiment analysis
- `--config PATH`: Path to configuration file
- `--log-level LEVEL`: Logging level (DEBUG|INFO|WARNING|ERROR|CRITICAL)

**Examples:**
```bash
# Demo mode
python main.py --demo

# Full analysis with 1000 samples
python main.py --sample-size 1000

# Advanced analysis with BERT
python main.py --use-bert --sample-size 500

# Custom configuration
python main.py --config custom_config.yaml --log-level DEBUG
```

---

## Configuration Schema

### YAML Configuration Structure

```yaml
# Data Configuration
data:
  raw_data_dir: "data/raw"
  processed_data_dir: "data/processed"
  sample_size: null
  demo_mode: false

# Analysis Configuration
analysis:
  sentiment:
    use_transformers: false
    text_column: "comments"
    preprocessing:
      remove_stopwords: true
      lowercase: true
      remove_special_chars: true
  satisfaction:
    weights:
      sentiment_score: 0.5
      review_count: 0.2
      price_inverse: 0.2
      availability: 0.1

# Forecasting Configuration
forecasting:
  periods: 180
  confidence_interval: 0.95
  polynomial_degree: 2

# Visualization Configuration
visualization:
  style: "whitegrid"
  figure_size: [10, 6]
  font_size: 12
  dpi: 300
  color_palette: "viridis"

# Output Configuration
output:
  plots_dir: "outputs/visualizations"
  reports_dir: "outputs/reports"
  models_dir: "outputs/models"
  logs_dir: "outputs/logs"
```

---

## Error Handling

### Common Exceptions

#### `FileNotFoundError`
Raised when required data files are missing.

**Solution:** Ensure CSV files are placed in the correct data directory.

#### `KeyError`
Raised when expected columns are missing from datasets.

**Solution:** Verify data file format matches expected schema.

#### `RuntimeError`
Raised when NLTK resources cannot be downloaded.

**Solution:** Check internet connection or manually download NLTK data.

#### `MemoryError`
Raised when dataset is too large for available memory.

**Solution:** Use `--sample-size` to reduce dataset size.

---

## Performance Considerations

### Memory Usage

- **VADER only**: ~250MB for 1K reviews
- **With BERT**: ~1GB+ for 1K reviews
- **Large datasets**: Use sample_size parameter

### Processing Time

- **VADER**: ~0.3 seconds per 100 reviews
- **BERT**: ~3 seconds per 100 reviews
- **Forecasting**: ~1 second regardless of size

### Optimization Tips

1. Use `sample_size` for development and testing
2. Disable BERT (`use_transformers=False`) for faster processing
3. Increase `chunk_size` in config for large datasets
4. Use SSD storage for better I/O performance

---

This API reference provides comprehensive documentation for all public interfaces in the project. For implementation details, refer to the source code and inline documentation. 