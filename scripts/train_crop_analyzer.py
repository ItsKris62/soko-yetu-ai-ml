import argparse
from app.models.crop_analyzer import CropAnalyzer
from app.utils.image_loader import PlantVillageLoader
from app.utils.logger import setup_logging
import os

def train_model(data_dir="data/raw/plantvillage", model_save_path="models/crop_analyzer.h5", epochs=30):
    """Train the crop analysis model"""
    try:
        # Load data
        loader = PlantVillageLoader(data_dir)
        train_gen, val_gen = loader.get_data_generators()
        
        # Train model
        model = CropAnalyzer()
        history = model.train(train_gen, val_gen, epochs=epochs)
        
        # Save model
        model.save(model_save_path)
        print(f"Model trained and saved to {model_save_path}")
        print(f"Validation accuracy: {max(history.history['val_accuracy'])}")
        
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