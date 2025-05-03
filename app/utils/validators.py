from pydantic import HttpUrl
import re
from app.utils.logger import logger

class Validators:
    @staticmethod
    def validate_image_url(image_url: HttpUrl) -> bool:
        """Validate that image URL is from trusted source"""
        trusted_domains = [
            "res.cloudinary.com",
            "sokoyetu.africa",
            "localhost"
        ]
        
        if not any(domain in str(image_url) for domain in trusted_domains):
            logger.warning(f"Untrusted image URL: {image_url}")
            raise ValueError("Image URL must be from trusted source")
        return True

    @staticmethod
    def validate_price_inputs(price_data: dict) -> bool:
        """Validate price prediction inputs"""
        if price_data.get('quantity', 0) <= 0:
            raise ValueError("Quantity must be positive")
            
        valid_units = ["kg", "g", "ton", "lb", "bag"]
        if price_data.get('unit', '').lower() not in valid_units:
            raise ValueError(f"Unit must be one of {valid_units}")
            
        return True

    @staticmethod
    def validate_country(country_id: int) -> bool:
        """Validate country exists"""
        # In practice, query database
        valid_countries = [1, 2, 3]  # Kenya, Uganda, Tanzania
        if country_id not in valid_countries:
            raise ValueError("Invalid country ID")
        return True