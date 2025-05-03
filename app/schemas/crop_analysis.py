from pydantic import BaseModel, HttpUrl
from typing import Optional
from datetime import datetime
from .base import BaseSchema

class CropAnalysisRequest(BaseSchema):
    image_url: HttpUrl
    category_id: int
    name: Optional[str] = None
    country_id: int
    county_id: Optional[int] = None
    sub_county_id: Optional[int] = None

class CropAnalysisResponse(BaseSchema):
    health_score: float
    pest_detected: bool
    disease_detected: bool
    crop_type: str
    confidence: float
    analysis_date: datetime
    recommendations: list[str]