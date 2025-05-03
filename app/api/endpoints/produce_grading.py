from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, HttpUrl
from datetime import datetime
from typing import Optional, Literal
import logging

from app.models.produce_grader import ProduceGrader
from app.services.image_service import process_produce_image
from app.utils.validators import validate_image_url
from app.utils.logger import logger

router = APIRouter()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

Grade = Literal["A", "B", "C", "D", "E"]

class ProduceGradingRequest(BaseModel):
    image_url: HttpUrl
    category_id: int
    product_name: str
    country_id: int
    county_id: Optional[int] = None

class ProduceGradingResponse(BaseModel):
    grade: Grade
    confidence: float
    grading_date: datetime
    defects: list[str]
    quality_attributes: dict  # size, color, shape scores

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return api_key

@router.post("/", response_model=ProduceGradingResponse)
async def grade_produce(
    request: ProduceGradingRequest,
    api_key: str = Depends(get_api_key)
):
    try:
        # Validate inputs
        validate_image_url(request.image_url)
        
        # Process image
        processed_image = await process_produce_image(request.image_url)
        
        # Load model
        grader = ProduceGrader("models/produce_grader.h5")
        
        # Perform grading
        grading_result = grader.grade(
            processed_image,
            request.category_id
        )
        
        logger.info(
            f"Produce grading completed for {request.product_name}",
            extra={
                "category_id": request.category_id,
                "grade": grading_result["grade"],
                "confidence": grading_result["confidence"]
            }
        )
        
        return ProduceGradingResponse(
            grade=grading_result["grade"],
            confidence=grading_result["confidence"],
            grading_date=datetime.utcnow(),
            defects=grading_result.get("defects", []),
            quality_attributes={
                "size": 0.8,
                "color": 0.9,
                "shape": 0.7
            }  # Example attributes
        )
        
    except Exception as e:
        logger.error(
            f"Produce grading failed: {str(e)}",
            exc_info=True,
            extra={
                "product_name": request.product_name,
                "image_url": request.image_url
            }
        )
        raise HTTPException(
            status_code=500,
            detail=f"Produce grading failed: {str(e)}"
        )