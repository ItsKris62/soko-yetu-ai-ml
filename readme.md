# Soko Yetu AI/ML Microservice

![Soko Yetu Logo](https://via.placeholder.com/150) <!-- Replace with actual logo URL -->

The **Soko Yetu AI/ML Microservice** is a FastAPI-based service designed to provide AI-driven insights for the Soko Yetu agricultural marketplace, serving farmers and buyers in East Africa (Kenya, Uganda, Tanzania). This microservice handles **crop analysis**, **price prediction**, **produce grading**, and **crop yield forecasting**, integrating with an Express.js backend and CockroachDB database. It leverages machine learning models (TensorFlow, scikit-learn) and computer vision (OpenCV) to deliver actionable insights for agricultural products.

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Setup Instructions (Windows)](#setup-instructions-windows)
- [Usage](#usage)
- [Training Models](#training-models)
- [Testing](#testing)
- [Deployment](#deployment)
- [Integration with Backend](#integration-with-backend)
- [Contributing](#contributing)
- [License](#license)

## Project Overview
Soko Yetu is a mobile-first, low-bandwidth-optimized platform connecting East African farmers and buyers. The AI/ML microservice enhances the platform by providing:
- **Crop Analysis**: Analyzes crop images to assess health, type, or quality.
- **Price Prediction**: Predicts market prices based on product details, location, and historical data.
- **Produce Grading**: Assigns quality grades (e.g., Grade A, Grade B) to produce.
- **Crop Yield Forecasting**: Forecasts crop yields using historical data, weather, and location.

The microservice exposes API endpoints for the Express.js backend to call, storing results in a CockroachDB database. It is designed for scalability, security, and performance in low-bandwidth environments.

## Features
- **Crop Analysis** (`POST /api/analyze-crop`):
  - Analyzes Cloudinary-hosted crop images for health, type, or pest detection.
  - Uses a convolutional neural network (CNN) trained on agricultural datasets.
- **Price Prediction** (`POST /api/predict-price`):
  - Predicts market prices using a Random Forest regression model.
  - Considers product type, location, quantity, and historical market data.
- **Produce Grading** (`POST /api/grade-produce`):
  - Assigns quality grades to produce based on image analysis.
  - Uses a CNN with a classification head.
- **Crop Yield Forecasting** (`POST /api/forecast-yield`):
  - Predicts crop yields using time-series or regression models.
  - Incorporates weather, location, and historical yield data.
- **Security**:
  - API key authentication for all endpoints.
  - Input validation to prevent injection attacks.
- **Integration**:
  - Seamless integration with Express.js backend and Cloudinary for image storage.
  - Stores results in CockroachDB (`products`, `crop_yield_forecasts`, `ai_insights` tables).
- **Performance**:
  - Optimized for low-latency inference.
  - Supports caching (Redis) for frequent predictions.

## Tech Stack
- **Backend**: FastAPI (Python)
- **Machine Learning**: TensorFlow (CNN, LSTM), scikit-learn (Random Forest)
- **Image Processing**: OpenCV, Pillow
- **Image Storage**: Cloudinary
- **Database**: CockroachDB (via SQLAlchemy, if needed)
- **Data Processing**: Pandas, NumPy
- **Deployment**: Gunicorn, Uvicorn, Docker
- **Testing**: pytest
- **Environment**: python-dotenv
- **Monitoring**: Logging (Python `logging`)


## Setup Instructions (Windows)
Follow these steps to set up the project on a Windows PC.

### Prerequisites
- Python 3.9+
- Docker Desktop
- Git
- Node.js (for Express.js backend integration, optional)
- Cloudinary account (for image storage)
- CockroachDB instance (optional, if querying directly)

### Steps
1. **Clone the Repository**:
   ```bash
   - git clone https://github.com/your-repo/soko-yetu-ai-ml.git
   - cd soko-yetu-ai-ml

   Set Up a Python Virtual Environment:
bash

python -m venv venv
.\venv\Scripts\activate

Install Dependencies:
bash

pip install -r requirements.txt

## Configure Environment Variables:
### Create a .env file in the project root:

### Run the FastAPI Development Server:
-  bash

- uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload



