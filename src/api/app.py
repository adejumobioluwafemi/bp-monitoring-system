from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd
import numpy as np
import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from model_loader import model_loader

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Hypertension Risk Prediction API",
    description="API for predicting hypertension risk based on clinical parameters",
    version="1.0.0"
)

class PatientData(BaseModel):
    Age: float
    BMI: float
    Systolic_BP: float
    Diastolic_BP: float
    Heart_Rate: float
    Gender: str  # "Male" or "Female"
    Medical_History: str  # "Yes" or "No"
    Smoking: str  # "Yes" or "No"
    Sporting: str  # "Yes" or "No"

class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    message: str


def interpret_prediction(prediction: int, probability: float) -> dict:
    """Interpret the prediction results"""
    risk_levels = {
        0: "Low Risk",
        1: "High Risk"
    }
    
    messages = {
        0: "Based on the provided information, you have a low risk of hypertension. Maintain a healthy lifestyle with regular checkups.",
        1: "Based on the provided information, you have a high risk of hypertension. Please consult with a healthcare professional for further evaluation and guidance."
    }
    
    return {
        "risk_level": risk_levels.get(prediction, "Unknown"),
        "message": messages.get(prediction, "Please consult a healthcare professional."),
        "confidence": f"{probability:.1%}"
    }

@app.get("/")
async def root():
    return {
        "message": "Hypertension Risk Prediction API",
        "status": "active",
        "version": "1.0.0",
        "model_loaded": model_loader.model is not None,
        "model_type": model_loader.model_type,
        "expected_features": model_loader.expected_features
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_status = "loaded" if model_loader.model is not None else "not loaded"
    
    return {
        "status": "healthy" if model_loader.model is not None else "degraded",
        "model_status": model_status,
        "model_type": model_loader.model_type,
        "expected_features": model_loader.expected_features,
        "timestamp": pd.Timestamp.now().isoformat()
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_hypertension_risk(patient_data: PatientData):
    """
    Predict hypertension risk based on patient data
    """
    try:
        if model_loader.model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Preprocess input data
        processed_input = model_loader.preprocess_input(patient_data)
        
        # Make prediction
        try:
            # Try predict_proba first (for sklearn models)
            if hasattr(model_loader.model, 'predict_proba'):
                probabilities = model_loader.model.predict_proba(processed_input)
                prediction = model_loader.model.predict(processed_input)
                probability = float(probabilities[0][1])  # Probability of class 1 (high risk)
            else:
                # Fallback: if model doesn't have predict_proba
                prediction = model_loader.model.predict(processed_input)
                probability = 0.8 if prediction[0] == 1 else 0.2
            
        except Exception as e:
            logger.error(f"Prediction method error: {e}")
            # Try alternative prediction approach
            prediction = model_loader.model.predict(processed_input)
            probability = 0.8 if prediction[0] == 1 else 0.2
        
        # Get prediction and probability
        predicted_class = int(prediction[0])
        
        # Interpret results
        interpretation = interpret_prediction(predicted_class, probability)
        
        logger.info(f"Prediction: {predicted_class}, Probability: {probability:.3f}")
        
        return PredictionResponse(
            prediction=predicted_class,
            probability=probability,
            risk_level=interpretation["risk_level"],
            message=interpretation["message"]
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/model-info")
async def get_model_info():
    """Get information about the loaded model"""
    if model_loader.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        model_info = {
            "model_type": model_loader.model_type,
            "expected_features": model_loader.expected_features,
            "has_predict_proba": hasattr(model_loader.model, 'predict_proba'),
        }
        
        # Add pipeline info if it's a pipeline
        if hasattr(model_loader.model, 'named_steps'):
            model_info["pipeline_steps"] = list(model_loader.model.named_steps.keys())
        
        return model_info
        
    except Exception as e:
        return {
            "model_type": model_loader.model_type,
            "error": str(e)
        }

@app.get("/debug-features")
async def debug_features():
    """Debug endpoint to see what features are being created"""
    sample_data = PatientData(
        Age=45,
        BMI=25.0,
        Systolic_BP=120,
        Diastolic_BP=80,
        Heart_Rate=72,
        Gender="Female",
        Medical_History="No",
        Smoking="No",
        Sporting="Yes"
    )
    
    try:
        processed = model_loader.preprocess_input(sample_data)
        return {
            "input_sample": sample_data.model_dump(),
            "processed_columns": list(processed.columns),
            "processed_values": processed.iloc[0].to_dict(),
            "expected_features": model_loader.expected_features
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)