from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base
from sqlalchemy.sql import func
import pandas as pd
import os
from app.utils.logger import logger

Base = declarative_base()

class Country(Base):
    __tablename__ = 'countries'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)

class County(Base):
    __tablename__ = 'counties'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), nullable=False)
    country_id = Column(Integer, ForeignKey('countries.id', ondelete='CASCADE'))

class Category(Base):
    __tablename__ = 'categories'
    id = Column(Integer, primary_key=True)
    name = Column(String(50), unique=True, nullable=False)
    description = Column(String)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class Product(Base):
    __tablename__ = 'products'
    id = Column(Integer, primary_key=True)
    farmer_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'))
    name = Column(String(100), nullable=False)
    price = Column(Float, nullable=False)
    image_url = Column(String(255))
    country_id = Column(Integer, ForeignKey('countries.id'))
    county_id = Column(Integer, ForeignKey('counties.id'))
    category_id = Column(Integer, ForeignKey('categories.id', ondelete='SET NULL'))
    ai_suggested_price = Column(Float)
    ai_quality_grade = Column(String(50))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class CropYieldForecast(Base):
    __tablename__ = 'crop_yield_forecasts'
    id = Column(Integer, primary_key=True)
    user_id = Column(Integer, ForeignKey('users.id', ondelete='CASCADE'))
    product_id = Column(Integer, ForeignKey('products.id', ondelete='SET NULL'))
    category_id = Column(Integer, ForeignKey('categories.id', ondelete='SET NULL'))
    forecasted_yield = Column(Float)
    confidence_score = Column(Float)
    forecast_date = Column(DateTime, nullable=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

class DataService:
    def __init__(self):
        self.db_url = os.getenv("DATABASE_URL", "cockroachdb://user:password@localhost:26257/soko_yetu")
        self.engine = create_engine(self.db_url)
        self.Session = sessionmaker(bind=self.engine)
        Base.metadata.create_all(self.engine)

    def get_historical_prices(self, category_id, country_id, county_id=None):
        """Get historical price data for price prediction"""
        try:
            session = self.Session()
            
            query = session.query(
                Product.price,
                Product.created_at
            ).filter(
                Product.category_id == category_id,
                Product.country_id == country_id
            )
            
            if county_id:
                query = query.filter(Product.county_id == county_id)
                
            prices = query.order_by(Product.created_at.desc()).limit(365).all()
            
            if not prices:
                return {
                    'mean': 0,
                    'min': 0,
                    'max': 0,
                    'count': 0
                }
                
            price_values = [p[0] for p in prices if p[0] is not None]
            
            return {
                'mean': float(np.mean(price_values)),
                'min': float(np.min(price_values)),
                'max': float(np.max(price_values)),
                'count': len(price_values),
                'latest': float(prices[0][0]),
                'latest_date': prices[0][1].isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to fetch historical prices: {str(e)}")
            return {
                'mean': 0,
                'min': 0,
                'max': 0,
                'count': 0
            }
        finally:
            session.close()

    def save_ai_insights(self, product_id, insights):
        """Save AI insights to database"""
        try:
            session = self.Session()
            
            # Update product with AI insights
            product = session.query(Product).get(product_id)
            if product:
                if 'ai_suggested_price' in insights:
                    product.ai_suggested_price = insights['ai_suggested_price']
                if 'ai_quality_grade' in insights:
                    product.ai_quality_grade = insights['ai_quality_grade']
                session.commit()
            
            # Save to ai_insights table
            if 'forecasted_yield' in insights:
                forecast = CropYieldForecast(
                    product_id=product_id,
                    forecasted_yield=insights['forecasted_yield'],
                    confidence_score=insights.get('confidence_score', 0),
                    forecast_date=insights.get('forecast_date', func.now())
                )
                session.add(forecast)
                session.commit()
                
        except Exception as e:
            logger.error(f"Failed to save AI insights: {str(e)}")
            session.rollback()
            raise
        finally:
            session.close()