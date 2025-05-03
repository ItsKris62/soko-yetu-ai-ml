from pydantic import BaseModel, HttpUrl
from datetime import datetime
from typing import Literal
from .base import BaseSchema

Grade = Literal["A", "B", "C", "D", "E"]

class ProduceGradingRequest(BaseSchema):
    image_url: HttpUrl
    category_id: int
    product_name: str
    country_id: int
    county_id: Optional[int] = None

class ProduceGradingResponse(BaseSchema):
    grade: Grade
    confidence: float
    grading_date: datetime
    defects: list[str]
    quality_attributes: dict