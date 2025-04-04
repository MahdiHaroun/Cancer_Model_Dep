from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import numpy as np
import pandas as pd 

model = joblib.load('model.pkl')
ss = joblib.load('scaler.pkl')
feature_names = joblib.load('feature_names.pkl')
app = FastAPI() 

app = FastAPI()



class CancerInput(BaseModel):
    radius_mean: float
    texture_mean: float
    perimeter_mean: float
    area_mean: float
    smoothness_mean: float
    compactness_mean: float
    concavity_mean: float
    concave_points_mean: float
    symmetry_mean: float
    fractal_dimension_mean: float
    radius_se: float
    texture_se: float
    perimeter_se: float
    area_se: float
    smoothness_se: float
    compactness_se: float
    concavity_se: float
    concave_points_se: float
    symmetry_se: float
    fractal_dimension_se: float
    radius_worst: float
    texture_worst: float
    perimeter_worst: float
    area_worst: float
    smoothness_worst: float
    compactness_worst: float
    concavity_worst: float
    concave_points_worst: float
    symmetry_worst: float
    fractal_dimension_worst: float


@app.post("/predict")
def predict_cancer(data: CancerInput):
    input_data = [getattr(data, feat) for feat in feature_names]
    input_scaled = ss.transform([input_data])
    prediction = model.predict(input_scaled)[0]
    result = "Malignant" if prediction == 1 else "Benign"
    return {"prediction": result}