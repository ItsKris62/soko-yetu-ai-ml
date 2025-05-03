from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from typing import Optional
from pydantic import BaseModel, HttpUrl
from datetime import datetime
import logging

from app.models.crop_analyzer import CropAnalyzer
from app.services.image_service import process_crop_image
from app.services.data_service import get_category_details
from app.utils.validators import validate_image_url
from app.utils.logger import logger

router = APIRouter()

# API Key Security
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class CropAnalysisRequest(BaseModel):
    image_url: HttpUrl
    category_id: int
    name: Optional[str] = None
    country_id: int
    county_id: Optional[int] = None
    sub_county_id: Optional[int] = None

class CropAnalysisResponse(BaseModel):
    health_score: float
    pest_detected: bool
    disease_detected: bool
    crop_type: str
    confidence: float
    analysis_date: datetime
    recommendations: list[str]

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return api_key

@router.post("/", response_model=CropAnalysisResponse)
async def analyze_crop(
    request: CropAnalysisRequest,
    api_key: str = Depends(get_api_key)
):
    try:
        # Validate inputs
        validate_image_url(request.image_url)
        
        # Get category details if needed
        category = await get_category_details(request.category_id)
        
        # Process image
        processed_image = await process_crop_image(request.image_url)
        
        # Load model (cached in practice)
        analyzer = CropAnalyzer("models/crop_analyzer.h5")
        
        # Perform analysis
        analysis_result = analyzer.analyze(
            processed_image,
            category.get("expected_types", []),
            request.country_id
        )
        
        logger.info(
            f"Crop analysis completed for {request.image_url}",
            extra={
                "category_id": request.category_id,
                "country_id": request.country_id,
                "result": analysis_result
            }
        )
        
        return CropAnalysisResponse(
            **analysis_result,
            analysis_date=datetime.utcnow(),
            recommendations=[
                "Increase watering frequency",
                "Apply organic fertilizer"
            ]  # Example recommendations
        )
        
    except Exception as e:
        logger.error(
            f"Crop analysis failed: {str(e)}",
            exc_info=True,
            extra={
                "image_url": request.image_url,
                "category_id": request.category_id
            }
        )
        raise HTTPException(
            status_code=500,
            detail=f"Crop analysis failed: {str(e)}"
        )