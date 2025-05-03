import argparse
import pandas as pd
from app.models.price_predictor import PricePredictor
from app.utils.logger import setup_logging
from sqlalchemy import create_engine
import os

def train_model(data_path=None, model_save_path="models/price_predictor.pkl"):
    """Train the price prediction model"""
    try:
        # Connect to database if no local path provided
        if data_path is None:
            engine = create_engine(os.getenv("DATABASE_URL"))
            query = "SELECT * FROM processed_price_data"
            data = pd.read_sql(query, engine)
        else:
            data = pd.read_csv(data_path)
        
        # Prepare features and target
        X = data[['category_id', 'quantity', 'country_id', 'county_id', 'season', 'month']]
        y = data['price']
        
        # Train model
        predictor = PricePredictor()
        metrics = predictor.train(X, y)
        
        # Save model
        predictor.save(model_save_path)
        print(f"Model trained with metrics: {metrics}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default=None,
                       help="Path to processed price data CSV")
    parser.add_argument("--model_path", type=str,
                       default="models/price_predictor.pkl",
                       help="Path to save trained model")
    
    args = parser.parse_args()
    setup_logging()
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    train_model(args.data_path, args.model_path)