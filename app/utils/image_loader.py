import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
from app.utils.logger import logger

class PlantVillageLoader:
    def __init__(self, data_dir="data/raw/plantvillage"):
        self.data_dir = data_dir
        self.train_dir = os.path.join(data_dir, "train")
        self.test_dir = os.path.join(data_dir, "test")
        
    def get_data_generators(self, target_size=(256, 256), batch_size=32):
        """Create image data generators"""
        try:
            train_datagen = ImageDataGenerator(
                rescale=1./255,
                rotation_range=40,
                width_shift_range=0.2,
                height_shift_range=0.2,
                shear_range=0.2,
                zoom_range=0.2,
                horizontal_flip=True,
                fill_mode='nearest',
                validation_split=0.2
            )
            
            test_datagen = ImageDataGenerator(rescale=1./255)
            
            train_generator = train_datagen.flow_from_directory(
                self.train_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='sparse',
                subset='training'
            )
            
            val_generator = train_datagen.flow_from_directory(
                self.train_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='sparse',
                subset='validation'
            )
            
            test_generator = test_datagen.flow_from_directory(
                self.test_dir,
                target_size=target_size,
                batch_size=batch_size,
                class_mode='sparse'
            )
            
            logger.info(f"Loaded {train_generator.samples} training images")
            logger.info(f"Loaded {val_generator.samples} validation images")
            logger.info(f"Loaded {test_generator.samples} test images")
            
            return train_generator, val_generator, test_generator
            
        except Exception as e:
            logger.error(f"Failed to load image data: {str(e)}")
            raise

    def get_class_names(self):
        """Get list of class names from directory structure"""
        try:
            classes = sorted(os.listdir(self.train_dir))
            logger.info(f"Found {len(classes)} classes")
            return classes
        except Exception as e:
            logger.error(f"Failed to get class names: {str(e)}")
            raise