from pydantic import BaseModel
from datetime import datetime
from .base import BaseSchema

class PricePredictionRequest(BaseSchema):
    category_id: int
    product_id: Optional[int] = None
    product_name: str
    quantity: float
    unit: str
    country_id: int
    county_id: Optional[int] = None
    sub_county_id: Optional[int] = None
    season: Optional[str] = None

class PricePredictionResponse(BaseSchema):
    predicted_price: float
    currency: str
    confidence: float
    prediction_date: datetime
    price_range: dict
    market_trend: str