import sys
from pathlib import Path # For robust path manipulation

import pandas as pd
import numpy as np
from datetime import datetime
import os
from sqlalchemy import create_engine
from PIL import Image
import cv2


from dotenv import load_dotenv


# Add the project root directory to sys.path
# This allows Python to find the 'app' module
project_root_dir = Path(__file__).resolve().parent.parent
if str(project_root_dir) not in sys.path: # Avoids adding duplicates
    sys.path.insert(0, str(project_root_dir))

# Load environment variables from .env file in the project root
# This should be done before trying to access os.getenv("DATABASE_URL")
dotenv_path = project_root_dir / ".env"
if dotenv_path.exists():
    load_dotenv(dotenv_path)
  # Loger.info

from app.utils.logger import logger, setup_logging
from app.utils.image_loader import PlantVillageLoader


# Call setup_logging() to configure the logger as defined in app/utils/logger.py
# Do this early, so all subsequent log messages use the intended configuration.
setup_logging()

# Log after logger is set up
if dotenv_path.exists():
    logger.info(f"Successfully loaded environment variables from: {dotenv_path}")


class DataPreprocessor:
    def __init__(self, db_url=None):
        self.db_url = db_url or os.getenv("DATABASE_URL")
        if not self.db_url:
            error_message = (
                "Database URL not configured. "
                "Please set the DATABASE_URL environment variable "
                "or pass a db_url argument to DataPreprocessor."
            )
            logger.error(error_message)
            raise ValueError(error_message)
        
        logger.info(f"Attempting to connect to database with URL: {self.db_url}")
        self.engine = create_engine(self.db_url)

    def preprocess_price_data(self, raw_path, output_path):
        """Preprocess price data for price prediction"""
        try:
            df = pd.read_csv(raw_path)
            
            # Read CSV handling multi-level header and footer
            # Use engine='python' because skipfooter is not supported by c engine
            df = pd.read_csv(raw_path, header=[0, 1], skiprows=0, skipfooter=1, engine='python')

            # Clean up multi-index column names
            # Combine Year and Month, handle the unnamed level_1 entries
            new_cols = []
            for year, month in df.columns:
                if 'Unnamed:' in year: # Handle the first column 'CROP'
                     new_cols.append('Crop')
                elif 'Unnamed:' in month: # Handle cases where month might be missing under year
                    new_cols.append(f'{year}_UnknownMonth')
                else:
                    new_cols.append(f'{year}_{month}')
            df.columns = new_cols

            # Convert to string first to handle potential non-string types, then clean
            df['Crop'] = df['Crop'].astype(str).str.replace(r'\.+', '', regex=True).str.strip()

            # Reshape data from wide to long format
            df = pd.melt(df, id_vars=['Crop'], var_name='Year_Month', value_name='Price')

            # Extract Year and Month
            df[['Year', 'Month_Str']] = df['Year_Month'].str.split('_', expand=True)
            month_map = {'Mar': 3, 'Sept': 9} # Add other months if they appear
            df['Month'] = df['Month_Str'].map(month_map)

            # Convert types
            df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
            df['Price'] = pd.to_numeric(df['Price'], errors='coerce')

            # Clean and feature engineer
            
            df = df.dropna(subset=['Price', 'Year', 'Month']) # Drop rows with missing essential info
            df['Season'] = df['Month'].apply(
                lambda x: 'rainy' if x in [3, 4, 5, 10, 11] else 'dry'
            )
            
            

            # Select and reorder columns for clarity (optional)
            df = df[['Crop', 'Year', 'Month', 'Season', 'Price']]
            
            # Save to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Price data preprocessed and saved to {output_path}")
            
            # Save to database
            df.to_sql('processed_price_data', self.engine, if_exists='replace', index=False)
            logger.info("Price data saved to database")
            
            return df
            
        except Exception as e:
            logger.error(f"Price data preprocessing failed: {str(e)}", exc_info=True)
            raise

    def preprocess_yield_data(self, raw_path, output_path):
        """Preprocess yield data for yield forecasting"""
        try:
            df = pd.read_csv(raw_path)
            
            # Clean and transform
            df = df.dropna(subset=['Value', 'Year'])
            df['date'] = pd.to_datetime(df['Year'], format='%Y')
            df = df.sort_values('date')
            df = df.rename(columns={'Year': 'original_year'}) # Rename the original 'Year' column
            
            
            # Add rolling features
            df['yield_3yr_avg'] = df['Value'].rolling(window=3).mean()
            df['yield_5yr_avg'] = df['Value'].rolling(window=5).mean()
            df['month'] = df['date'].dt.month
            df['year'] = df['date'].dt.year
            
            # Save to file
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            logger.info(f"Yield data preprocessed and saved to {output_path}")
            
            # Save to database
            df.to_sql('processed_yield_data', self.engine, if_exists='replace', index=False)
            logger.info("Yield data saved to database")
            
            return df
            
        except Exception as e:
            logger.error(f"Yield data preprocessing failed: {str(e)}", exc_info=True)
            raise

    def preprocess_image_data(self, raw_dir, output_dir, target_size=(256, 256), dataset_type="crop"):
        """Preprocess image data for crop analysis or produce grading"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            class_dirs = [d for d in os.listdir(raw_dir) if os.path.isdir(os.path.join(raw_dir, d))]
            
            for class_name in class_dirs:
                class_path = os.path.join(raw_dir, class_name)
                output_class_path = os.path.join(output_dir, class_name)
                os.makedirs(output_class_path, exist_ok=True)
                
                for img_name in os.listdir(class_path):
                    img_path = os.path.join(class_path, img_name)
                    try:
                        img = cv2.imread(img_path)
                        if img is None:
                            continue
                        img = cv2.resize(img, target_size)
                        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        output_path = os.path.join(output_class_path, img_name)
                        cv2.imwrite(output_path, img)
                    except Exception as e:
                        logger.warning(f"Failed to process image {img_path}: {str(e)}")
                        continue
            
            logger.info(f"Image data preprocessed for {dataset_type} and saved to {output_dir}")
            
            # Save metadata to database (optional)
            metadata = pd.DataFrame({
                'class_name': class_dirs,
                'dataset_type': dataset_type,
                'image_count': [len(os.listdir(os.path.join(output_dir, c))) for c in class_dirs]
            })
            metadata.to_sql(f'processed_{dataset_type}_image_metadata', self.engine, if_exists='replace', index=False)
            logger.info(f"{dataset_type} image metadata saved to database")
            
        except Exception as e:
            logger.error(f"Image data preprocessing failed: {str(e)}", exc_info=True)
            raise

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    
    # Preprocess price data
    preprocessor.preprocess_price_data(
        raw_path="data/raw/market_prices/average-retail-market-prices-of-selected-food-crops.csv",
        output_path="data/processed/price_data.csv"
    )
    
    # Preprocess yield data
    preprocessor.preprocess_yield_data(
        raw_path="data/raw/faostat/FAOSTAT_data_crop-products_4-20-2025.csv",
        output_path="data/processed/yield_data.csv"
    )
    
    # Preprocess crop analysis images
    preprocessor.preprocess_image_data(
        raw_dir="data/raw/plantvillage",
        output_dir="data/processed/plantvillage",
        target_size=(256, 256),
        dataset_type="crop"
    )
    
    # Preprocess produce grading images
    preprocessor.preprocess_image_data(
        raw_dir="data/raw/produce_grading",
        output_dir="data/processed/produce_grading",
        target_size=(512, 512),
        dataset_type="produce"
    )