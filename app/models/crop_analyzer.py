import tensorflow as tf
import numpy as np
from app.utils.mlflow_tracker import MLflowTracker
from app.utils.logger import logger
from typing import Dict, List

class CropAnalyzer:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.tracker = MLflowTracker()
        self.class_names = [
            "healthy", "pest_infected", "disease_infected", 
            "nutrient_deficient", "water_stressed"
        ]
        self.crop_types = {
            1: "Maize", 2: "Wheat", 3: "Tomato", 
            4: "Potato", 5: "Beans"
        }
        
        if model_path:
            try:
                self.model = tf.keras.models.load_model(model_path)
                logger.info(f"Model loaded from {model_path}")
            except Exception as e:
                logger.error(f"Failed to load model: {str(e)}")
                self.model = self._build_model()
        else:
            self.model = self._build_model()

    def _build_model(self):
        """Build CNN model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(256,256,3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(len(self.class_names), activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def train(self, train_data, val_data, epochs=30):
        """Train the model"""
        callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
            tf.keras.callbacks.ModelCheckpoint(
                self.model_path if self.model_path else 'best_model.h5',
                save_best_only=True
            )
        ]
        
        history = self.model.fit(
            train_data,
            validation_data=val_data,
            epochs=epochs,
            callbacks=callbacks
        )
        
        # Log to MLflow
        self.tracker.log_training(
            model=self.model,
            model_type='tensorflow',
            params={
                'model_name': 'crop_analyzer',
                'epochs': epochs,
                'input_shape': (256,256,3)
            },
            metrics={
                'val_accuracy': max(history.history['val_accuracy']),
                'val_loss': min(history.history['val_loss'])
            }
        )
        
        return history

    def analyze(self, image: np.ndarray, expected_types: List[str] = None, country_id: int = None) -> Dict:
        """Analyze crop health from image"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        predictions = self.model.predict(image)
        pred_class = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        
        crop_type = self._predict_crop_type(image)
        
        if expected_types and crop_type not in expected_types:
            crop_type = f"Possibly {crop_type} (unexpected for category)"
        
        result = {
            'health_score': 1 - float(predictions[0][pred_class]),
            'pest_detected': 'pest' in self.class_names[pred_class],
            'disease_detected': 'disease' in self.class_names[pred_class],
            'crop_type': crop_type,
            'confidence': confidence
        }
        
        self.tracker.log_prediction(
            model_name='crop_analyzer',
            input_data={'image_shape': image.shape},
            prediction=result
        )
        
        return result

    def _predict_crop_type(self, image: np.ndarray) -> str:
        """Predict crop type (simplified example)"""
        return "Maize"  # In production, replace with actual model

    def save(self, path: str):
        """Save model to path"""
        self.model.save(path)