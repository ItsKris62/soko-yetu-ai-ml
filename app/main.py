from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from prometheus_fastapi_instrumentator import Instrumentator
from app.api.endpoints import (
    crop_analysis,
    price_prediction,
    produce_grading,
    yield_forecast
)
from app.utils.logger import setup_logging
import os

# Initialize logging
setup_logging()

app = FastAPI(
    title="Soko Yetu AI/ML Microservice",
    description="Agricultural AI Services for East Africa",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routers
app.include_router(
    crop_analysis.router,
    prefix="/api/v1/analyze-crop",
    tags=["Crop Analysis"]
)
app.include_router(
    price_prediction.router,
    prefix="/api/v1/predict-price",
    tags=["Price Prediction"]
)
app.include_router(
    produce_grading.router,
    prefix="/api/v1/grade-produce",
    tags=["Produce Grading"]
)
app.include_router(
    yield_forecast.router,
    prefix="/api/v1/forecast-yield",
    tags=["Yield Forecasting"]
)

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

@app.on_event("startup")
async def startup_event():
    """Initialize services on startup"""
    # Initialize database connection
    from app.services.data_service import DataService
    DataService()  # Creates tables if they don't exist

@app.get("/")
async def root():
    return {
        "message": "Soko Yetu AI Service",
        "status": "operational",
        "version": "1.0.0"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy"}