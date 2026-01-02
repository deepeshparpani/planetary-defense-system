from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Planetary Defense API")

# Load the Pipeline (which includes SMOTE logic and the XGB model)
MODEL_PATH = "models/neo_classifier.joblib"
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
else:
    model = None

class AsteroidRequest(BaseModel):
    est_diameter_min: float
    relative_velocity: float
    miss_distance: float
    absolute_magnitude: float

def calculate_features(df):
    """Must match the logic used in train.py exactly"""
    df['size_dist_ratio'] = df['est_diameter_min'] / (df['miss_distance'] + 1e-5)
    df['kinetic_proxy'] = (df['relative_velocity']**2) * df['est_diameter_min']
    df['velocity_dist_ratio'] = df['relative_velocity'] / (df['miss_distance'] + 1e-5)
    return df

@app.post("/predict")
def predict(data: AsteroidRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded.")
    
    # Convert input to DataFrame
    input_df = pd.DataFrame([data.dict()])
    
    # Apply Feature Engineering
    input_df = calculate_features(input_df)
    
    # The pipeline handles the prediction
    prediction = int(model.predict(input_df)[0])
    probability = float(model.predict_proba(input_df)[0][1])
    
    return {
        "is_hazardous": bool(prediction),
        "probability": f"{probability:.2%}"
    }