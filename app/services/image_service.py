import cv2
import numpy as np
import cloudinary
import cloudinary.uploader
from io import BytesIO
from urllib.request import urlopen
from PIL import Image
import os
from app.utils.logger import logger

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
    api_key=os.getenv("CLOUDINARY_API_KEY"),
    api_secret=os.getenv("CLOUDINARY_API_SECRET")
)

class ImageService:
    @staticmethod
    def process_crop_image(image_url):
        """Process crop image for analysis"""
        try:
            resp = urlopen(image_url)
            img = Image.open(BytesIO(resp.read()))
            img_array = np.array(img)
            
            if img_array.shape[2] == 3:
                img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
            
            img_array = cv2.resize(img_array, (256, 256))
            img_array = img_array / 255.0
            
            return img_array
            
        except Exception as e:
            logger.error(f"Failed to process crop image: {str(e)}")
            raise

    @staticmethod
    def process_produce_image(image_url):
        """Process produce image for grading"""
        try:
            transformed_url = cloudinary.utils.cloudinary_url(
                image_url,
                width=512,
                height=512,
                crop="fill",
                quality="auto",
                format="jpg"
            )[0]
            
            resp = urlopen(transformed_url)
            img = Image.open(BytesIO(resp.read()))
            img_array = np.array(img)
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
            img_array = cv2.GaussianBlur(img_array, (5, 5), 0)
            
            return img_array
            
        except Exception as e:
            logger.error(f"Failed to process produce image: {str(e)}")
            raise

    @staticmethod
    def upload_image(file_path, folder="soko_yetu"):
        """Upload image to Cloudinary"""
        try:
            response = cloudinary.uploader.upload(
                file_path,
                folder=folder,
                quality="auto",
                format="jpg"
            )
            return response['secure_url']
        except Exception as e:
            logger.error(f"Image upload failed: {str(e)}")
            raise