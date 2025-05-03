from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel
from datetime import datetime
from typing import Optional
import logging
import pandas as pd
import numpy as np

from app.models.price_predictor import PricePredictor
from app.services.data_service import get_historical_prices
from app.utils.validators import validate_price_inputs
from app.utils.logger import logger

router = APIRouter()

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class PricePredictionRequest(BaseModel):
    category_id: int
    product_id: Optional[int] = None
    product_name: str
    quantity: float
    unit: str  # kg, g, ton, etc.
    country_id: int
    county_id: Optional[int] = None
    sub_county_id: Optional[int] = None
    season: Optional[str] = None  # rainy, dry, etc.

class PricePredictionResponse(BaseModel):
    predicted_price: float
    currency: str
    confidence: float
    prediction_date: datetime
    price_range: dict  # min, max
    market_trend: str  # increasing, decreasing, stable

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(
            status_code=401,
            detail="Invalid API Key"
        )
    return api_key

@router.post("/", response_model=PricePredictionResponse)
async def predict_price(
    request: PricePredictionRequest,
    api_key: str = Depends(get_api_key)
):
    try:
        # Validate inputs
        validate_price_inputs(request)
        
        # Get historical data
        historical_data = await get_historical_prices(
            request.category_id,
            request.country_id,
            request.county_id
        )
        
        # Convert to features DataFrame
        features = pd.DataFrame([{
            "category_id": request.category_id,
            "quantity": request.quantity,
            "country_id": request.country_id,
            "county_id": request.county_id or 0,
            "season": request.season or "unknown",
            "month": datetime.now().month
        }])
        
        # Load model
        predictor = PricePredictor("models/price_predictor.pkl")
        
        # Make prediction
        prediction = predictor.predict(features)
        
        # Get additional market context
        market_trend = "increasing" if prediction > historical_data["mean"] else "decreasing"
        
        logger.info(
            f"Price prediction completed for {request.product_name}",
            extra={
                "category_id": request.category_id,
                "predicted_price": prediction,
                "market_trend": market_trend
            }
        )
        
        return PricePredictionResponse(
            predicted_price=float(prediction),
            currency="KES",  # Default to Kenyan Shilling
            confidence=0.85,  # Example value
            prediction_date=datetime.utcnow(),
            price_range={
                "min": float(historical_data["min"]),
                "max": float(historical_data["max"])
            },
            market_trend=market_trend
        )
        
    except Exception as e:
        logger.error(
            f"Price prediction failed: {str(e)}",
            exc_info=True,
            extra={
                "product_name": request.product_name,
                "category_id": request.category_id
            }
        )
        raise HTTPException(
            status_code=500,
            detail=f"Price prediction failed: {str(e)}"
        )