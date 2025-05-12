import argparse
import pandas as pd
import os
import sys # Required for sys.path manipulation
from pathlib import Path # For robust path manipulation
from sqlalchemy import create_engine

# Add the project root directory to sys.path
# This allows Python to find the 'app' module
project_root_dir = Path(__file__).resolve().parent.parent
if str(project_root_dir) not in sys.path: # Avoids adding duplicates
    sys.path.insert(0, str(project_root_dir))

from app.models.price_predictor import PricePredictor
from app.utils.logger import setup_logging
from app.utils.mlflow_tracker import MLflowTracker

def train_model(data_path=None, model_save_path="models/price_predictor.pkl"):
    """Train the price prediction model"""
    try:
        # Initialize MLflow tracking
        tracker = MLflowTracker()

        # Connect to database if no local path provided
        if data_path is None:
            engine = create_engine(os.getenv("DATABASE_URL"))
            query = "SELECT * FROM processed_price_data"
            data = pd.read_sql(query, engine)
        else:
            data = pd.read_csv(data_path)

        # Prepare features and target
        feature_columns = ['category_id', 'quantity', 'country_id', 'county_id', 'season', 'month']
        X = data[feature_columns]
        y = data['price']

        # Train model
        predictor = PricePredictor()
        metrics = predictor.train(X, y)

        # Save model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        predictor.save(model_save_path)

        # Log metrics to MLflow
        # Assuming predictor.model gives access to the underlying scikit-learn model
        tracker.log_training(
            model=predictor.model, # Or however the scikit-learn model is accessed
            model_type='sklearn',
            params={
                'model_name': 'price_predictor',
                'features': feature_columns
            },
            metrics=metrics
        )

        print(f"Model trained with metrics: {metrics}")
        print(f"Model saved to {model_save_path}")
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