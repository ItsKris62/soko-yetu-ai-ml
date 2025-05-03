import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
from app.utils.mlflow_tracker import MLflowTracker
from app.utils.logger import logger
from typing import Dict, Union
from datetime import datetime

class YieldForecaster:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.tracker = MLflowTracker()
        self.features = [
            'category_id', 'country_id', 'county_id',
            'month', 'year', 'historical_yield_3yr_avg'
        ]
        
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        if model_path:
            try:
                self.model = load(model_path)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")

    def train(self, X: pd.DataFrame, y: pd.Series) -> Dict:
        """Train the yield forecasting model"""
        self.model.fit(X, y)
        
        metrics = {
            'train_r2': self.model.score(X, y)
        }
        
        self.tracker.log_training(
            model=self.model,
            model_type='sklearn',
            params={
                'model_name': 'yield_forecaster',
                'n_estimators': 100
            },
            metrics=metrics
        )
        
        if self.model_path:
            self.save(self.model_path)
        
        return metrics

    def forecast(self, features: Union[pd.DataFrame, Dict]) -> Dict:
        """Make yield forecast"""
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        for col in self.features:
            if col not in features.columns:
                features[col] = 0
                
        prediction = self.model.predict(features)
        forecast = max(0, float(prediction[0]))
        
        result = {
            'forecasted_yield': forecast,
            'confidence': 0.85,  # Placeholder
            'forecast_date': datetime.now().isoformat()
        }
        
        self.tracker.log_prediction(
            model_name='yield_forecaster',
            input_data=features.to_dict(),
            prediction=result
        )
        
        return result

    def save(self, path: str):
        """Save model to path"""
        dump(self.model, path)