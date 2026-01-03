from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os

app = FastAPI(title="Planetary Defense API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load Model
MODEL_PATH = "models/neo_classifier.joblib"
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"Error loading model: {e}")

class AsteroidRequest(BaseModel):
    est_diameter_min: float
    relative_velocity: float
    miss_distance: float
    absolute_magnitude: float

# Root endpoint (helps verify basic connectivity)
@app.get("/")
def root():
    return {"message": "Planetary Defense API is active"}

# Health check endpoint (Fixes the 404 error)
@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": model is not None}

@app.post("/predict")
def predict(data: AsteroidRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded on server.")
    
    try:
        input_df = pd.DataFrame([data.dict()])
        # Engineering logic
        input_df['size_dist_ratio'] = input_df['est_diameter_min'] / (input_df['miss_distance'] + 1e-5)
        input_df['kinetic_proxy'] = (input_df['relative_velocity']**2) * input_df['est_diameter_min']
        input_df['velocity_dist_ratio'] = input_df['relative_velocity'] / (input_df['miss_distance'] + 1e-5)
        
        prediction = int(model.predict(input_df)[0])
        probability = float(model.predict_proba(input_df)[0][1])
        
        return {
            "is_hazardous": bool(prediction),
            "probability": f"{probability:.2%}"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))