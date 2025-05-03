from fastapi import APIRouter, HTTPException, Depends, Security
from fastapi.security import APIKeyHeader
from datetime import datetime
from typing import Optional
from app.schemas.yield_forecast import YieldForecastRequest, YieldForecastResponse
from app.models.yield_forecaster import YieldForecaster
from app.services.data_service import DataService
from app.utils.validators import Validators
from app.utils.logger import logger
import os

router = APIRouter()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

async def get_api_key(api_key: str = Security(api_key_header)):
    if api_key != os.getenv("API_KEY"):
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return api_key

@router.post("/", response_model=YieldForecastResponse)
async def forecast_yield(
    request: YieldForecastRequest,
    api_key: str = Depends(get_api_key)
):
    try:
        # Validate inputs
        Validators.validate_country(request.country_id)
        
        # Prepare features
        features = {
            'category_id': request.category_id,
            'country_id': request.country_id,
            'county_id': request.county_id or 0,
            'month': request.forecast_date.month,
            'year': request.forecast_date.year,
            'historical_yield_3yr_avg': DataService().get_historical_yield_avg(
                request.category_id, 
                request.country_id,
                request.county_id
            )
        }
        
        # Make forecast
        forecaster = YieldForecaster("models/yield_forecaster.pkl")
        forecast = forecaster.forecast(features)
        
        # Save to database
        DataService().save_yield_forecast(
            user_id=request.user_id,
            product_id=request.product_id,
            category_id=request.category_id,
            forecast_data=forecast
        )
        
        return YieldForecastResponse(
            forecasted_yield=forecast['forecasted_yield'],
            confidence_score=forecast['confidence'],
            forecast_date=datetime.utcnow(),
            recommendations=[
                "Consider rotating crops next season",
                "Increase irrigation during dry spells"
            ]
        )
        
    except Exception as e:
        logger.error(f"Yield forecast failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))