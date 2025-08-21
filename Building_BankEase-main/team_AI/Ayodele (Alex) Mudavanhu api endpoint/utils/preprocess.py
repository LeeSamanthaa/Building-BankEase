# utils/preprocess.py

import pandas as pd
import joblib
import os

# Load all encoders and transformers
MODEL_DIR = "models"

def load_model(name):
    return joblib.load(os.path.join(MODEL_DIR, name))

# Load label encoders
encoders = {
    "TransactionType": load_model("label_encoder_TransactionType.pkl"),
    "Location": load_model("label_encoder_Location.pkl"),
    "Occupation": load_model("label_encoder_CustomerOccupation.pkl"),
    "Channel": load_model("label_encoder_Channel.pkl")  # in case needed later
}

# Load scalers and transformers
scaler = load_model("customer_profile_scaler.pkl")
svd = load_model("svd_model.pkl")

# -------------------
# Fraud Detection Preprocessing
# -------------------

def preprocess_fraud_input(data: dict) -> pd.DataFrame:
    """
    Preprocesses fraud detection input for prediction.
    Required fields in data:
        - transaction_amount
        - transaction_type
        - account_age_days
        - location
    """
    df = pd.DataFrame([data])
    df["transaction_type"] = encoders["TransactionType"].transform(df["transaction_type"])
    df["location"] = encoders["Location"].transform(df["location"])
    return df

# -------------------
# Customer Clustering Preprocessing
# -------------------

def preprocess_customer_input(data: dict):
    """
    Preprocesses customer input for clustering.
    Required fields in data:
        - age
        - income
        - occupation
        - location
    Returns a transformed NumPy array ready for KMeans.
    """
    df = pd.DataFrame([data])
    df["occupation"] = encoders["Occupation"].transform(df["occupation"])
    df["location"] = encoders["Location"].transform(df["location"])
    scaled = scaler.transform(df)
    reduced = svd.transform(scaled)
    return reduced
