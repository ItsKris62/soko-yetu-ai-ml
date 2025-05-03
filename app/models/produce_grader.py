import tensorflow as tf
import numpy as np
from typing import Dict, Literal
from app.utils.mlflow_tracker import MLflowTracker
from app.utils.logger import logger

Grade = Literal["A", "B", "C", "D", "E"]

class ProduceGrader:
    def __init__(self, model_path=None):
        self.model_path = model_path
        self.tracker = MLflowTracker()
        
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
        """Build grading model architecture"""
        model = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(512,512,3)),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dense(5, activation='softmax')  # 5 grades
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model

    def grade(self, image: np.ndarray, category_id: int) -> Dict:
        """Grade produce quality"""
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)
            
        predictions = self.model.predict(image)
        grade_idx = np.argmax(predictions, axis=1)[0]
        confidence = float(np.max(predictions))
        
        grades = ["A", "B", "C", "D", "E"]
        result = {
            'grade': grades[grade_idx],
            'confidence': confidence,
            'defects': self._detect_defects(image),
            'quality_attributes': {
                'size': 0.8,  # Placeholder
                'color': 0.9,
                'shape': 0.7
            }
        }
        
        self.tracker.log_prediction(
            model_name='produce_grader',
            input_data={'image_shape': image.shape, 'category_id': category_id},
            prediction=result
        )
        
        return result

    def _detect_defects(self, image: np.ndarray) -> List[str]:
        """Detect defects in produce (simplified example)"""
        return []  # In production, implement actual defect detection

    def save(self, path: str):
        """Save model to path"""
        self.model.save(path)