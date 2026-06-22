from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import joblib

app = FastAPI(title="Fraud Detection API", version="1.0")

# ======================
# Load artifacts
# ======================
model = joblib.load("model.pkl")
threshold = joblib.load("threshold.pkl")
features = joblib.load("features.pkl")


# ======================
# Input schema
# ======================
class Transaction(BaseModel):
    merchant: str
    category: str
    gender: str
    city: str
    state: str
    job: str

    amt: float
    zip: int
    lat: float
    long: float
    city_pop: int
    unix_time: int
    merch_lat: float
    merch_long: float
    age: int
    hour: int
    day: int
    month: int
    dayofweek: int


# ======================
# Health check
# ======================
@app.get("/")
def home():
    return {"message": "Fraud Detection API is running 🚀"}


# ======================
# Prediction endpoint
# ======================
@app.post("/predict")
def predict(data: Transaction):

    # Convert input to DataFrame
    df = pd.DataFrame([data.dict()])

    # Ensure correct column order
    df = df.reindex(columns=features)

    # Predict probability
    prob = model.predict_proba(df)[0][1]

    # Apply threshold
    pred = int(prob > threshold)

    return {
        "fraud_probability": float(prob),
        "fraud_prediction": pred,
        "threshold_used": float(threshold)
    }