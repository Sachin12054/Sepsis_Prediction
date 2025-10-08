
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np
from typing import List
import json

app = FastAPI(title="Sepsis Prediction API", version="1.0.0")

# Load model and metadata
model = joblib.load("models/sepsis_model.pkl")
with open("models/model_metadata.json", "r") as f:
    model_metadata = json.load(f)

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    sepsis_probability: float
    sepsis_prediction: int
    risk_level: str
    model_version: str

@app.post("/predict", response_model=PredictionResponse)
async def predict_sepsis(request: PredictionRequest):
    try:
        # Validate input
        if len(request.features) != len(model_metadata["features"]):
            raise HTTPException(status_code=400, 
                              detail=f"Expected {len(model_metadata['features'])} features, got {len(request.features)}")
        
        # Create DataFrame with proper feature names
        features_df = pd.DataFrame([request.features], columns=model_metadata["features"])
        
        # Make prediction
        probability = model.predict_proba(features_df)[0, 1]
        prediction = int(model.predict(features_df)[0])
        
        # Determine risk level
        if probability >= 0.8:
            risk_level = "HIGH"
        elif probability >= 0.5:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
        
        return PredictionResponse(
            sepsis_probability=float(probability),
            sepsis_prediction=prediction,
            risk_level=risk_level,
            model_version=model_metadata["version"]
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy", "model_version": model_metadata["version"]}

@app.get("/model/info")
async def model_info():
    return model_metadata

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
