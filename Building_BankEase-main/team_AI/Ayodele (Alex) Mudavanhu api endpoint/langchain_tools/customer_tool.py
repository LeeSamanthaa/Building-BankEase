from langchain.tools import tool
import requests

@tool
def get_customer_cluster(age: int, income: float, occupation: str, location: str) -> str:
    """Get customer cluster for profiling and recommendation."""
    url = "http://localhost:8000/customer-cluster"
    payload = {
        "age": age,
        "income": income,
        "occupation": occupation,
        "location": location
    }
    response = requests.post(url, json=payload)
    return response.json()
