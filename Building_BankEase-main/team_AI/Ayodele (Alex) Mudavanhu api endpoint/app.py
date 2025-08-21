# app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib

from utils.preprocess import preprocess_fraud_input, preprocess_customer_input

app = FastAPI(title="ML Model API")

# Load Models and Preprocessors
fraud_model = joblib.load("models/fraud_detection_model.pkl")
scaler = joblib.load("models/customer_profile_scaler.pkl")
kmeans = joblib.load("models/kmeans_cluster_model.pkl")
svd = joblib.load("models/svd_model.pkl")

label_encoders = {
    "Channel": joblib.load("models/label_encoder_Channel.pkl"),
    "Occupation": joblib.load("models/label_encoder_CustomerOccupation.pkl"),
    "Location": joblib.load("models/label_encoder_Location.pkl"),
    "TransactionType": joblib.load("models/label_encoder_TransactionType.pkl"),
}

# Request Models
class FraudInput(BaseModel):
    transaction_amount: float
    transaction_type: str
    account_age_days: int
    location: str

class CustomerInput(BaseModel):
    age: int
    income: float
    occupation: str
    location: str

@app.post("/predict-fraud")
def predict_fraud(data: FraudInput):
    try:
        df = pd.DataFrame([data.dict()])
        df["transaction_type"] = label_encoders["TransactionType"].transform(df["transaction_type"])
        df["location"] = label_encoders["Location"].transform(df["location"])
        pred = fraud_model.predict(df)[0]
        return {"is_fraud": bool(pred)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/customer-cluster")
def customer_cluster(data: CustomerInput):
    try:
        df = pd.DataFrame([data.dict()])
        df["occupation"] = label_encoders["Occupation"].transform(df["occupation"])
        df["location"] = label_encoders["Location"].transform(df["location"])
        scaled = scaler.transform(df)
        reduced = svd.transform(scaled)
        cluster = kmeans.predict(reduced)[0]
        return {"customer_cluster": int(cluster)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.post("/predict-fraud")
def predict_fraud(data: FraudInput):
    try:
        df = preprocess_fraud_input(data.dict())
        prediction = fraud_model.predict(df)[0]
        return {"is_fraud": bool(prediction)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/customer-cluster")
def customer_cluster(data: CustomerInput):
    try:
        transformed = preprocess_customer_input(data.dict())
        cluster = kmeans.predict(transformed)[0]
        return {"customer_cluster": int(cluster)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

