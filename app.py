"""
Task 5: Model Serving using FastAPI
-----------------------------------
REST API exposing the trained Titanic Random Forest classification pipeline.
Endpoints:
  - GET /        : API overview and documentation links.
  - GET /health  : Health check returning API status and model state.
  - POST /predict: Predicts survival for a passenger with Pydantic validation.
"""

from contextlib import asynccontextmanager
from datetime import datetime
import os
import time
from typing import Optional

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import pandas as pd
import joblib

MODEL_FILE = "model.pkl"
model_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global model_pipeline
    if not os.path.exists(MODEL_FILE):
        raise RuntimeError(f"Trained model '{MODEL_FILE}' not found. Please run train.py first.")
    print(f"[*] Loading model pipeline from '{MODEL_FILE}'...")
    model_pipeline = joblib.load(MODEL_FILE)
    print("[+] Model pipeline loaded successfully into memory.")
    yield
    print("[*] Shutting down API service...")


app = FastAPI(
    title="Titanic Survival Prediction API",
    description="MLOps Model Serving REST API for Titanic Disaster Survival Prediction.",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Enable CORS for cross-origin client support
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic Schemas for Request & Response validation
class PassengerInput(BaseModel):
    Pclass: int = Field(
        ...,
        ge=1,
        le=3,
        description="Ticket class: 1 = 1st Class, 2 = 2nd Class, 3 = 3rd Class",
        json_schema_extra={"example": 1}
    )
    Sex: str = Field(
        ...,
        pattern="^(male|female)$",
        description="Passenger gender: 'male' or 'female'",
        json_schema_extra={"example": "female"}
    )
    Age: Optional[float] = Field(
        None,
        ge=0.0,
        le=120.0,
        description="Age in years (imputed if null)",
        json_schema_extra={"example": 29.0}
    )
    SibSp: int = Field(
        0,
        ge=0,
        description="Number of siblings / spouses aboard",
        json_schema_extra={"example": 0}
    )
    Parch: int = Field(
        0,
        ge=0,
        description="Number of parents / children aboard",
        json_schema_extra={"example": 0}
    )
    Fare: Optional[float] = Field(
        None,
        ge=0.0,
        description="Passenger fare paid in GBP",
        json_schema_extra={"example": 78.50}
    )
    Embarked: Optional[str] = Field(
        "S",
        pattern="^(C|Q|S)$",
        description="Port of Embarkation: C = Cherbourg, Q = Queenstown, S = Southampton",
        json_schema_extra={"example": "S"}
    )


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_type: str
    timestamp: str
    version: str


class PredictionResponse(BaseModel):
    status: str
    prediction: int
    label: str
    probability_survived: float
    probability_not_survived: float
    confidence: float
    latency_ms: float
    input_data: dict


@app.get("/", tags=["General"])
def root():
    return {
        "message": "Welcome to Titanic ML Model Serving API",
        "documentation": "/docs",
        "health_endpoint": "/health",
        "predict_endpoint": "/predict"
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    """
    Check the health status of the API and verify that the ML model pipeline is loaded.
    """
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model is not loaded."
        )
    return HealthResponse(
        status="healthy",
        model_loaded=True,
        model_type="RandomForestClassifier (Scikit-Learn Pipeline)",
        timestamp=datetime.now().isoformat(),
        version="1.0.0"
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict_survival(passenger: PassengerInput):
    """
    Accept passenger details, validate data types and constraints via Pydantic,
    pass features through the complete preprocessing pipeline, and return predicted survival.
    """
    if model_pipeline is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Model pipeline is currently unavailable."
        )

    # Convert incoming validated Pydantic model to Pandas DataFrame
    input_dict = passenger.model_dump()
    df_input = pd.DataFrame([input_dict])

    # Run inference & compute latency
    start_time = time.perf_counter()
    prediction_raw = model_pipeline.predict(df_input)[0]
    probabilities = model_pipeline.predict_proba(df_input)[0]
    latency_ms = (time.perf_counter() - start_time) * 1000

    pred_int = int(prediction_raw)
    label = "Survived" if pred_int == 1 else "Did Not Survive"
    confidence = round(float(probabilities[pred_int] * 100), 2)

    return PredictionResponse(
        status="success",
        prediction=pred_int,
        label=label,
        probability_survived=round(float(probabilities[1]), 4),
        probability_not_survived=round(float(probabilities[0]), 4),
        confidence=confidence,
        latency_ms=round(latency_ms, 2),
        input_data=input_dict
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
