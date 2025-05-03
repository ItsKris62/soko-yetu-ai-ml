from pydantic import BaseModel
from datetime import datetime
from .base import BaseSchema

class YieldForecastRequest(BaseSchema):
    user_id: int
    category_id: int
    product_id: Optional[int] = None
    country_id: int
    county_id: Optional[int] = None
    forecast_date: datetime

class YieldForecastResponse(BaseSchema):
    forecasted_yield: float
    confidence_score: float
    forecast_date: datetime
    recommendations: list[str]