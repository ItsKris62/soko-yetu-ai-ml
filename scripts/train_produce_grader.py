import argparse
import os
from app.models.produce_grader import ProduceGrader
from app.utils.image_loader import PlantVillageLoader
from app.utils.logger import setup_logging
from app.utils.mlflow_tracker import MLflowTracker

def train_model(data_dir="data/processed/produce_grading", model_save_path="models/produce_grader.h5", epochs=20):
    """Train the produce grading model"""
    try:
        # Initialize MLflow tracking
        tracker = MLflowTracker()
        
        # Load data (reusing PlantVillageLoader with adjusted target size)
        loader = PlantVillageLoader(data_dir)
        train_gen, val_gen, _ = loader.get_data_generators(target_size=(512, 512), batch_size=16)
        
        # Initialize and train model
        model = ProduceGrader()
        history = model.train(train_gen, val_gen, epochs=epochs)
        
        # Save model
        os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
        model.save(model_save_path)
        
        # Log metrics to MLflow
        tracker.log_training(
            model=model.model,
            model_type='tensorflow',
            params={
                'model_name': 'produce_grader',
                'epochs': epochs,
                'input_shape': (512, 512, 3),
                'batch_size': 16
            },
            metrics={
                'val_accuracy': max(history.history['val_accuracy']),
                'val_loss': min(history.history['val_loss'])
            }
        )
        
        print(f"Model trained and saved to {model_save_path}")
        print(f"Best validation accuracy: {max(history.history['val_accuracy']):.4f}")
        
    except Exception as e:
        print(f"Training failed: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ProduceGrader model")
    parser.add_argument("--data_dir", type=str, default="data/processed/produce_grading",
                        help="Path to processed produce grading dataset")
    parser.add_argument("--model_path", type=str, default="models/produce_grader.h5",
                        help="Path to save trained model")
    parser.add_argument("--epochs", type=int, default=20,
                        help="Number of training epochs")
    
    args = parser.parse_args()
    setup_logging()
    
    train_model(args.data_dir, args.model_path, args.epochs)