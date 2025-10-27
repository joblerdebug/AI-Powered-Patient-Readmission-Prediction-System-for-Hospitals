from fastapi import FastAPI, HTTPException, Depends
from pydantic import BaseModel, validator
from typing import List, Optional, Dict, Any
import numpy as np
import joblib
import pandas as pd
from datetime import datetime
import logging
import uvicorn
from contextlib import asynccontextmanager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global variables for model and scaler
models = {}
scaler = None
feature_names = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load models on startup
    global models, scaler, feature_names
    
    try:
        # Load best model
        models['readmission'] = joblib.load('models/saved_models/best_model.joblib')
        
        # Load scaler and feature names (you'd need to save these during training)
        # For demo, we'll create a mock
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        
        # Mock feature names - in real scenario, save these during training
        feature_names = [f'feature_{i}' for i in range(20)]
        
        logger.info("Models loaded successfully")
    except Exception as e:
        logger.error(f"Error loading models: {e}")
        raise e
    
    yield  # Server runs here
    
    # Cleanup on shutdown
    models.clear()
    logger.info("Models unloaded")

app = FastAPI(
    title="Patient Readmission Prediction API",
    description="AI-powered API for predicting 30-day patient readmission risk",
    version="1.0.0",
    lifespan=lifespan
)

# Pydantic models for request/response validation
class PatientData(BaseModel):
    age: float
    blood_pressure: float
    cholesterol: float
    previous_admissions: int
    comorbidity_count: int
    medication_count: int
    lab_result_1: float
    lab_result_2: float
    # Add more features as needed
    
    @validator('age')
    def validate_age(cls, v):
        if v < 0 or v > 120:
            raise ValueError('Age must be between 0 and 120')
        return v
    
    @validator('blood_pressure')
    def validate_bp(cls, v):
        if v < 50 or v > 250:
            raise ValueError('Blood pressure must be between 50 and 250')
        return v

class PredictionRequest(BaseModel):
    patient_id: str
    patient_data: PatientData
    model_type: str = "readmission"

class PredictionResponse(BaseModel):
    patient_id: str
    prediction: int
    probability: float
    risk_level: str
    confidence: str
    timestamp: str
    model_version: str

class HealthCheck(BaseModel):
    status: str
    timestamp: str
    model_loaded: bool

# Utility functions
def preprocess_input(patient_data: PatientData) -> np.ndarray:
    """Preprocess incoming patient data for model prediction"""
    # Convert to dataframe
    input_dict = patient_data.dict()
    input_df = pd.DataFrame([input_dict])
    
    # Ensure correct feature order and missing features
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0  # or appropriate default value
    
    # Reorder columns to match training
    input_df = input_df[feature_names]
    
    # Scale features
    input_scaled = scaler.transform(input_df)
    
    return input_scaled

def calculate_risk_level(probability: float) -> str:
    """Convert probability to risk level"""
    if probability < 0.3:
        return "Low"
    elif probability < 0.7:
        return "Medium"
    else:
        return "High"

def calculate_confidence(probability: float) -> str:
    """Calculate prediction confidence"""
    distance_from_decision = abs(probability - 0.5)
    if distance_from_decision > 0.3:
        return "High"
    elif distance_from_decision > 0.15:
        return "Medium"
    else:
        return "Low"

# API endpoints
@app.get("/", response_model=HealthCheck)
async def health_check():
    """Health check endpoint"""
    return HealthCheck(
        status="healthy",
        timestamp=datetime.utcnow().isoformat(),
        model_loaded=len(models) > 0
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict_readmission_risk(request: PredictionRequest):
    """Predict patient readmission risk"""
    try:
        logger.info(f"Received prediction request for patient {request.patient_id}")
        
        # Check if model is loaded
        if request.model_type not in models:
            raise HTTPException(status_code=404, detail="Model not found")
        
        # Preprocess input
        processed_input = preprocess_input(request.patient_data)
        
        # Make prediction
        model = models[request.model_type]
        prediction = model.predict(processed_input)[0]
        probability = model.predict_proba(processed_input)[0][1]
        
        # Prepare response
        response = PredictionResponse(
            patient_id=request.patient_id,
            prediction=int(prediction),
            probability=float(probability),
            risk_level=calculate_risk_level(probability),
            confidence=calculate_confidence(probability),
            timestamp=datetime.utcnow().isoformat(),
            model_version="1.0.0"
        )
        
        logger.info(f"Prediction completed for patient {request.patient_id}: {response.risk_level} risk")
        
        return response
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model/info")
async def get_model_info():
    """Get information about loaded models"""
    model_info = {}
    
    for model_name, model in models.items():
        model_info[model_name] = {
            "model_type": type(model).__name__,
            "n_features": getattr(model, "n_features_in_", "Unknown"),
            "classes": getattr(model, "classes_", []).tolist() if hasattr(model, "classes_") else []
        }
    
    return model_info

# Additional endpoints for model management
@app.post("/model/retrain")
async def retrain_model():
    """Endpoint to trigger model retraining (simplified)"""
    # In production, this would trigger a retraining pipeline
    return {"message": "Retraining triggered", "status": "processing"}

if __name__ == "__main__":
    uvicorn.run(
        "fastapi_app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,  # Disable in production
        log_level="info"
    )
