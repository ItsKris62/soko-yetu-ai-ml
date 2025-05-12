import argparse
import sys # Required for sys.path manipulation
from pathlib import Path # For robust path manipulation

import os

# Add the project root directory to sys.path
# This allows Python to find the 'app' module
project_root_dir = Path(__file__).resolve().parent.parent
if str(project_root_dir) not in sys.path: # Avoids adding duplicates
    sys.path.insert(0, str(project_root_dir))

from app.models.crop_analyzer import CropAnalyzer
from app.utils.image_loader import PlantVillageLoader
from app.utils.logger import setup_logging
from app.utils.mlflow_tracker import MLflowTracker

def train_model(data_dir="data/raw/plantvillage", model_save_path="models/crop_analyzer.h5", epochs=30):
    """Train the crop analysis model"""
    try:
        # Load data
        loader = PlantVillageLoader(data_dir)
        train_gen, val_gen, test_gen = loader.get_data_generators()

        # Initialize MLflow tracking
        tracker = MLflowTracker()

        # Train model
        model = CropAnalyzer()
        history = model.train(train_gen, val_gen, epochs=epochs)

        # Save model
        model.save(model_save_path)
        print(f"Model trained and saved to {model_save_path}")

        val_accuracy = max(history.history['val_accuracy']) if 'val_accuracy' in history.history else 0.0
        val_loss = min(history.history['val_loss']) if 'val_loss' in history.history else float('inf')

        # Log metrics to MLflow
        # Assuming model.model gives access to the underlying Keras model
        # and train_gen.batch_size and model.model.input_shape are accessible
        tracker.log_training(
            model=model.model, # Or however the Keras model is accessed from CropAnalyzer instance
            model_type='tensorflow',
            params={
                'model_name': 'crop_analyzer',
                'epochs': epochs,
                'batch_size': train_gen.batch_size if hasattr(train_gen, 'batch_size') else 'N/A',
                'input_shape': model.model.input_shape[1:] if hasattr(model, 'model') and hasattr(model.model, 'input_shape') else 'N/A'
            },
            metrics={
                'val_accuracy': val_accuracy,
                'val_loss': val_loss
            }
        )
        print(f"Validation accuracy: {val_accuracy}")

    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, 
                       default="data/raw/plantvillage",
                       help="Path to PlantVillage dataset")
    parser.add_argument("--model_path", type=str,
                       default="models/crop_analyzer.h5",
                       help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=30,
                       help="Number of training epochs")
    
    args = parser.parse_args()
    setup_logging()
    
    # Ensure models directory exists
    os.makedirs(os.path.dirname(args.model_path), exist_ok=True)
    
    train_model(args.data_dir, args.model_path, args.epochs)