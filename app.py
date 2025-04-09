# import joblib
# import numpy as np
# import pandas as pd
# from fastapi import FastAPI, HTTPException, Request
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel, Field
# import logging

# # Set up logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Load saved models with error handling
# try:
#     sarimax_model = joblib.load("sarimax_model.joblib")
#     scaler = joblib.load("scaler.joblib")
#     label_encoders = joblib.load("label_encoders.joblib")
#     logger.info("Models loaded successfully.")
# except Exception as e:
#     logger.error(f"Error loading models: {e}")
#     raise Exception(f"Error loading models: {e}")

# # Get all crop names from the label encoder for the Crop column
# all_crops = label_encoders["Crop"].classes_

# # Initialize FastAPI app
# app = FastAPI(title="Crop Demand Prediction API")

# # Comprehensive CORS Configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=[
#         "http://localhost:3000",   # Next.js development server
#         "http://127.0.0.1:3000",   # Alternative localhost
#         "http://localhost:8000",    # Backend server
#         "http://127.0.0.1:8000",   # Another localhost
#         "*",                       # Allow all origins during development
#     ],
#     allow_credentials=True,
#     allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
#     allow_headers=["*"]   # Allows all headers
# )

# # Input Data Model with validation using Pydantic
# class InputData(BaseModel):
#     region: str = Field(..., description="Agricultural region")
#     crop: str = Field(..., description="Crop type")
#     production: float = Field(..., gt=0, description="Production quantity")
#     season: str = Field(..., description="Crop season")
#     soil_type: str = Field(..., description="Soil type")
#     months_ahead: int = Field(..., gt=0, le=12, description="Forecast period in months")

# # Prediction Route with Enhanced Error Handling
# @app.post("/predict")
# def predict_demand(data: InputData):
#     try:
#         # Convert months to days for SARIMAX (1 month = 30 days)
#         future_days = data.months_ahead * 30
        
#         # Encode categorical variables with error handling
#         encoded_input = {}
#         for col in ["Region", "Crop", "Season", "Soil_Type"]:
#             user_input_value = data.dict()[col.lower()]
            
#             if user_input_value not in label_encoders[col].classes_:
#                 logger.warning(f"Value '{user_input_value}' for column '{col}' not found in encoder classes. Falling back to '{label_encoders[col].classes_[0]}'")
#                 # Use first available category as fallback if not found
#                 user_input_value = label_encoders[col].classes_[0]
            
#             encoded_input[col] = label_encoders[col].transform([user_input_value])[0]
        
#         # Prepare input data for prediction (as a batch, even if it's one sample)
#         new_data = pd.DataFrame({
#             "Region": [encoded_input["Region"]],
#             "Crop": [encoded_input["Crop"]],
#             "Production": [data.production],
#             "Season": [encoded_input["Season"]],
#             "Soil_Type": [encoded_input["Soil_Type"]]
#         })
        
#         # Scale the numerical values based on the training scaler
#         new_data_scaled = scaler.transform(new_data)
        
#         # Reshape the exogenous variables to match the model's expected shape
#         # We need to match (180, 5) shape; for this, we need to repeat the single input for 180 times (as if it's a batch)
#         exog_input = np.repeat(new_data_scaled, 180, axis=0)
        
#         predictions = []
        
#         for crop in all_crops:
#             crop_encoded = label_encoders["Crop"].transform([crop])[0]
            
#             # Update the input data with the current crop for prediction
#             new_data["Crop"] = crop_encoded
#             new_data_scaled = scaler.transform(new_data)
#             exog_input = np.repeat(new_data_scaled, 180, axis=0)
            
#             # Predict demand using SARIMAX model
#             predicted_demand = sarimax_model.forecast(steps=future_days, exog=exog_input)[-1]
            
#             predictions.append({
#                 "crop": crop,
#                 "demand_forecast": float(predicted_demand)
#             })
        
#         # Sort predictions by demand forecast in descending order
#         sorted_predictions = sorted(predictions, key=lambda x: x["demand_forecast"], reverse=True)
        
#         return {"sorted_demand_forecast": sorted_predictions}
    
#     except Exception as e:
#         logger.error(f"Error during prediction: {str(e)}")
#         raise HTTPException(
#             status_code=500,
#             detail=f"Prediction failed: {str(e)}"
#         )


# # Health Check Route
# @app.get("/health")
# def health_check():
#     return {"status": "healthy"}

# # Run with: uvicorn main:app --reload

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load saved models with error handling
try:
    sarimax_model = joblib.load("sarimax_model.joblib")
    scaler = joblib.load("scaler.joblib")
    label_encoders = joblib.load("label_encoders.joblib")
    logger.info("Models loaded successfully.")
except Exception as e:
    logger.error(f"Error loading models: {e}")
    raise Exception(f"Error loading models: {e}")

# Get all crop names from the label encoder for the Crop column
all_crops = label_encoders["Crop"].classes_

# Initialize FastAPI app
app = FastAPI(title="Crop Demand Prediction API")

# Comprehensive CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",   # Next.js development server
        "http://127.0.0.1:3000",   # Alternative localhost
        "http://localhost:8000",    # Backend server
        "http://127.0.0.1:8000",   # Another localhost
        "*",                       # Allow all origins during development
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, etc.)
    allow_headers=["*"]   # Allows all headers
)

# Input Data Model with validation using Pydantic
class InputData(BaseModel):
    region: str = Field(..., description="Agricultural region")
    crop: str = Field(..., description="Crop type")
    production: float = Field(..., gt=0, description="Production quantity")
    season: str = Field(..., description="Crop season")
    soil_type: str = Field(..., description="Soil type")
    months_ahead: int = Field(..., gt=0, le=12, description="Forecast period in months")

# Prediction Route with Enhanced Error Handling
@app.post("/predict")
def predict_demand(data: InputData):
    try:
        # Convert months to days for SARIMAX (1 month = 30 days)
        future_days = data.months_ahead * 30
        
        # Encode categorical variables with error handling
        encoded_input = {}
        for col in ["Region", "Crop", "Season", "Soil_Type"]:
            user_input_value = data.dict()[col.lower()]
            
            if user_input_value not in label_encoders[col].classes_:
                logger.warning(f"Value '{user_input_value}' for column '{col}' not found in encoder classes. Falling back to '{label_encoders[col].classes_[0]}'")
                # Use first available category as fallback if not found
                user_input_value = label_encoders[col].classes_[0]
            
            encoded_input[col] = label_encoders[col].transform([user_input_value])[0]
        
        # Prepare input data for prediction (as a batch, even if it's one sample)
        new_data = pd.DataFrame({
            "Region": [encoded_input["Region"]],
            "Crop": [encoded_input["Crop"]],
            "Production": [data.production],
            "Season": [encoded_input["Season"]],
            "Soil_Type": [encoded_input["Soil_Type"]]
        })
        
        # Scale the numerical values based on the training scaler
        new_data_scaled = scaler.transform(new_data)
        
        # Initialize the list to store predictions
        predictions = []
        
        for crop in all_crops:
            crop_encoded = label_encoders["Crop"].transform([crop])[0]
            
            # Update the input data with the current crop for prediction
            new_data["Crop"] = crop_encoded
            new_data_scaled = scaler.transform(new_data)
            
            # Ensure the exogenous variables match the required shape (steps, features)
            exog_input = np.tile(new_data_scaled, (future_days, 1))  # Tile the data for the forecast period
            
            # Predict demand using SARIMAX model
            predicted_demand = sarimax_model.forecast(steps=future_days, exog=exog_input)[-1]
            
            predictions.append({
                "crop": crop,
                "demand_forecast": float(predicted_demand)
            })
        
        # Sort predictions by demand forecast in descending order
        sorted_predictions = sorted(predictions, key=lambda x: x["demand_forecast"], reverse=True)
        
        return {"sorted_demand_forecast": sorted_predictions}
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )

# Health Check Route
@app.get("/health")
def health_check():
    return {"status": "healthy"}

# Run with: uvicorn main:app --reload
