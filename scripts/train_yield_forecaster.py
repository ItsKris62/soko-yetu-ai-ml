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

from app.models.yield_forecaster import YieldForecaster
from app.utils.logger import setup_logging
from app.utils.mlflow_tracker import MLflowTracker

def train_model(data_path=None, model_save_path="models/yield_forecaster.pkl"):
    """Train the yield forecasting model"""
    try:
        # Initialize MLflow tracking
        tracker = MLflowTracker()
        
        # Load data
        if data_path is None:
            engine = create_engine(os.getenv("DATABASE_URL"))
            query = "SELECT * FROM processed_yield_data"
            data = pd.read_sql(query, engine)
        else:
            data = pd.read_csv(data_path)
        
        # Prepare features and target
        features = ['category_id', 'country_id', 'county_id', 'month', 'year', 'yield_3yr_avg']
        X = data[features].fillna(0)
        y = data['Value']
        
        # Train model
        forecaster = YieldForecaster()
        metrics = forecaster.train(X, y)
        
        # Save model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        forecaster.save(model_save_path)
        
        # Log metrics to MLflow
        tracker.log_training(
            model=forecaster.model,
            model_type='sklearn',
            params={
                'model_name': 'yield_forecaster',
                'n_estimators': 100,
                'features': features
            },
            metrics=metrics
        )
        
        print(f"Model trained with metrics: {metrics}")
        print(f"Model saved to {model_save_path}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YieldForecaster model")
    parser.add_argument("--data_path", type=str, default=None,
                        help="Path to processed yield data CSV")
    parser.add_argument("--model_path", type=str, default="models/yield_forecaster.pkl",
                        help="Path to save trained model")
    
    args = parser.parse_args()
    setup_logging()
    
    train_model(args.data_path, args.model_path)