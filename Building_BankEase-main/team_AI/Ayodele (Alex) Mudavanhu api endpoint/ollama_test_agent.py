# ollama_test_agent.py

from langchain.agents import initialize_agent, AgentType
from langchain_community.chat_models import ChatOllama
from langchain.tools import tool
import requests

# Define LangChain tool to call your local ML API
@tool
def detect_fraud(transaction_amount: float, transaction_type: str, account_age_days: int, location: str) -> dict:
    """Predict if a transaction is fraudulent using a local ML API."""
    url = "http://localhost:8000/predict-fraud"
    payload = {
        "transaction_amount": transaction_amount,
        "transaction_type": transaction_type,
        "account_age_days": account_age_days,
        "location": location
    }
    try:
        res = requests.post(url, json=payload)
        return res.json()
    except Exception as e:
        return {"error": str(e)}

# Use Ollama via LangChain
llm = ChatOllama(model="llama3", temperature=0)

# Initialize agent with your fraud detection tool
agent = initialize_agent(
    tools=[detect_fraud],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Ask the LLM to figure it out
#agent.run("Is a $5000 transfer from a 3-month-old account in Harare fraudulent?")
