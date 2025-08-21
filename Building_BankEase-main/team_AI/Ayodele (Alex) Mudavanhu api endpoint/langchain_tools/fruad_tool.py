from langchain.tools import tool
import requests

@tool
def detect_fraud(transaction_amount: float, transaction_type: str, account_age_days: int, location: str) -> str:
    """Detect whether a transaction is fraudulent."""
    url = "http://localhost:8000/predict-fraud"
    payload = {
        "transaction_amount": transaction_amount,
        "transaction_type": transaction_type,
        "account_age_days": account_age_days,
        "location": location
    }
    response = requests.post(url, json=payload)
    return response.json()

