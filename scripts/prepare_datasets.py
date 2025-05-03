import pandas as pd
import numpy as np
from datetime import datetime
import os
from sqlalchemy import create_engine
from app.utils.logger import logger

class DataPreprocessor:
    def __init__(self, db_url=None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        self.engine = create_engine(self.db_url)
        
    def prepare_price_data(self, raw_path, output_path):
        """Prepare price prediction dataset"""
        try:
            # Load and merge data
            df = pd.read_csv(raw_path)
            
            # Feature engineering
            df['month'] = datetime.now().month
            df['season'] = df['month'].apply(
                lambda x: 'rainy' if x in [3,4,5,10,11] else 'dry'
            )
            
            # Save processed data
            df.to_csv(output_path, index=False)
            logger.info(f"Price data prepared and saved to {output_path}")
            
            # Also save to database
            df.to_sql('processed_price_data', self.engine, if_exists='replace', index=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Price data preparation failed: {str(e)}", exc_info=True)
            raise

    def prepare_yield_data(self, raw_path, output_path):
        """Prepare yield forecasting dataset"""
        try:
            df = pd.read_csv(raw_path)
            
            # Clean and transform
            df['date'] = pd.to_datetime(df['Year'], format='%Y')
            df = df.sort_values('date')
            
            # Add rolling features
            df['yield_3yr_avg'] = df['Value'].rolling(3).mean()
            df['yield_5yr_avg'] = df['Value'].rolling(5).mean()
            
            # Save processed data
            df.to_csv(output_path, index=False)
            logger.info(f"Yield data prepared and saved to {output_path}")
            
            # Save to database
            df.to_sql('processed_yield_data', self.engine, if_exists='replace', index=False)
            
            return df
            
        except Exception as e:
            logger.error(f"Yield data preparation failed: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Prepare price data
    preprocessor.prepare_price_data(
        "data/raw/market_prices/average-retail-market-prices-of-selected-food-crops.csv",
        "data/processed/price_data.csv"
    )
    
    # Prepare yield data
    preprocessor.prepare_yield_data(
        "data/raw/faostat/FAOSTAT_data_crop-products_4-20-2025.csv",
        "data/processed/yield_data.csv"
    )