import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)

class SatisfactionForecaster:
    def __init__(self):
        """Initialize the forecaster"""
        self.model = LinearRegression()
        self.poly = PolynomialFeatures(degree=2, include_bias=False)
        
    def forecast_satisfaction(self, satisfaction_scores, periods=180):
        """Generate satisfaction forecast using polynomial regression"""
        try:
            # Extract overall satisfaction score
            if isinstance(satisfaction_scores, pd.DataFrame):
                ts_data = satisfaction_scores['satisfaction_score']
            else:
                ts_data = pd.Series(satisfaction_scores)
            
            # Prepare data for forecasting
            dates = pd.date_range(start='2023-01-01', periods=len(ts_data), freq='D')
            ts_data.index = dates
            
            # Create features (time index)
            X = np.arange(len(ts_data)).reshape(-1, 1)
            X_poly = self.poly.fit_transform(X)
            
            # Fit model
            self.model.fit(X_poly, ts_data.values)
            
            # Generate future dates and predictions
            future_X = np.arange(len(ts_data), len(ts_data) + periods).reshape(-1, 1)
            future_X_poly = self.poly.transform(future_X)
            forecast = self.model.predict(future_X_poly)
            
            # Calculate confidence intervals (using standard error)
            y_pred = self.model.predict(X_poly)
            mse = np.mean((ts_data.values - y_pred) ** 2)
            std_err = np.sqrt(mse)
            
            # Plot results
            plt.figure(figsize=(12, 6))
            plt.plot(ts_data.index, ts_data, label='Historical')
            
            future_dates = pd.date_range(start=dates[-1] + timedelta(days=1), periods=periods, freq='D')
            plt.plot(future_dates, forecast, 'r', label='Forecast')
            
            # Add confidence intervals
            plt.fill_between(future_dates,
                           forecast - 1.96 * std_err,
                           forecast + 1.96 * std_err,
                           color='r', alpha=0.1)
            
            plt.title('Satisfaction Score Forecast')
            plt.xlabel('Date')
            plt.ylabel('Satisfaction Score')
            plt.legend()
            plt.grid(True)
            plt.savefig('output/plots/satisfaction_forecast.png')
            plt.close()
            
            # Return forecast data
            forecast_data = {
                'dates': future_dates.strftime('%Y-%m-%d').tolist(),
                'forecast': forecast.tolist(),
                'lower_bound': (forecast - 1.96 * std_err).tolist(),
                'upper_bound': (forecast + 1.96 * std_err).tolist(),
                'mse': float(mse),
                'std_err': float(std_err)
            }
            
            return forecast_data
            
        except Exception as e:
            logger.error(f"Error in forecasting: {str(e)}")
            return None
    
    def plot_forecast(self, forecast_df, save_path):
        """Plot the forecast results"""
        try:
            fig = self.model.plot(forecast_df)
            plt.title('Satisfaction Score Forecast')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved forecast plot to {save_path}")
        except Exception as e:
            logger.error(f"Error plotting forecast: {e}")
    
    def plot_components(self, forecast_df, save_path):
        """Plot the forecast components"""
        try:
            fig = self.model.plot_components(forecast_df)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            plt.close()
            logger.info(f"Saved components plot to {save_path}")
        except Exception as e:
            logger.error(f"Error plotting components: {e}")
    
    def get_forecast_insights(self, forecast_df):
        """Generate insights from the forecast"""
        try:
            insights = {
                'trend': {
                    'current': float(forecast_df['trend'].iloc[-1]),
                    'direction': 'increasing' if forecast_df['trend'].diff().iloc[-1] > 0 else 'decreasing'
                },
                'forecast': {
                    'next_month': float(forecast_df['yhat'].tail(30).mean()),
                    'next_quarter': float(forecast_df['yhat'].tail(90).mean()),
                    'confidence': {
                        'lower': float(forecast_df['yhat_lower'].tail(30).mean()),
                        'upper': float(forecast_df['yhat_upper'].tail(30).mean())
                    }
                }
            }
            return insights
        except Exception as e:
            logger.error(f"Error generating forecast insights: {e}")
            return None 