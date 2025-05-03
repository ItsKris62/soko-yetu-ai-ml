from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from joblib import dump, load
from app.utils.mlflow_tracker import MLflowTracker
from app.utils.logger import logger
import pandas as pd
import numpy as np
from typing import Dict, Union

class PricePredictor:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.tracker = MLflowTracker()
        self.features = [
            'category_id', 'quantity', 'country_id', 
            'county_id', 'season', 'month'
        ]
        
        # Define preprocessing
        numeric_features = ['quantity', 'category_id', 'country_id', 'county_id', 'month']
        categorical_features = ['season']
        
        numeric_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])
        
        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        
        self.preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_features),
                ('cat', categorical_transformer, categorical_features)
            ])
        
        self.model = Pipeline(steps=[
            ('preprocessor', self.preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1))
        ])
        
        if model_path:
            try:
                self.model = load(model_path)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")

    def train(self, X: pd.DataFrame, y: pd.Series, test_data=None) -> Dict:
        """Train the model"""
        self.model.fit(X, y)
        
        metrics = {'train_r2': self.model.score(X, y)}
        
        if test_data:
            X_test, y_test = test_data
            metrics['test_r2'] = self.model.score(X_test, y_test)
        
        self.tracker.log_training(
            model=self.model,
            model_type='sklearn',
            params={
                'model_name': 'price_predictor',
                'n_estimators': 100
            },
            metrics=metrics
        )
        
        if self.model_path:
            self.save(self.model_path)
        
        return metrics

    def predict(self, features: Union[pd.DataFrame, Dict]) -> float:
        """Make price prediction"""
        if isinstance(features, dict):
            features = pd.DataFrame([features])
        
        for col in self.features:
            if col not in features.columns:
                features[col] = 0
                
        prediction = self.model.predict(features)
        price = max(0, float(prediction[0]))
        
        self.tracker.log_prediction(
            model_name='price_predictor',
            input_data=features.to_dict(),
            prediction={'predicted_price': price}
        )
        
        return price

    def save(self, path: str):
        """Save model to path"""
        dump(self.model, path)