import joblib
import pandas as pd
import os
import logging

# Setup basic logging to help you debug during development
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelManager:
    """
    Handles loading the trained model and performing inference.
    Separating this logic makes the FastAPI main.py cleaner and easier to test.
    """
    def __init__(self, model_path: str = "models/neo_classifier.joblib"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        """Loads the model from disk with error handling."""
        if not os.path.exists(self.model_path):
            logger.error(f"Model file not found at {self.model_path}. Did you run scripts/train.py?")
            return None
        
        try:
            model = joblib.load(self.model_path)
            logger.info("Successfully loaded NEO Classifier model.")
            return model
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return None

    def predict_hazard(self, input_features: dict):
        """
        Takes a dictionary of features and returns a prediction and probability.
        """
        if self.model is None:
            return {"error": "Model not available"}

        # Convert the dictionary into a DataFrame for the XGBoost model
        # The column names must match the order/names used during training
        df = pd.DataFrame([input_features])
        
        # Get binary prediction (0 or 1)
        prediction = int(self.model.predict(df)[0])
        
        # Get probability (specifically for the 'Hazardous' class)
        # Probabilities are returned as [prob_class_0, prob_class_1]
        probability = self.model.predict_proba(df)[0][1]

        return {
            "is_hazardous": bool(prediction),
            "hazard_probability": float(probability)
        }

# Instantiate a singleton instance to be used by the FastAPI app
neo_model_manager = ModelManager()