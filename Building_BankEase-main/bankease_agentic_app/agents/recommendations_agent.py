"""
Right now, running the script each time this agent is called
Since the response is static for a user, we can save the generated answer in session_state if customer returns back to this agent
This saves llm tokens
This file also uses response from insights agent. Can reuse that too.
"""

import streamlit as st
import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (roc_auc_score, f1_score, precision_score,recall_score, 
                            confusion_matrix, classification_report)
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from lightgbm import LGBMClassifier
from surprise import Dataset, Reader, SVD
from surprise.model_selection import cross_validate
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

""" from agents.prompts import INSIGHTS_PROMPT, RECOMMENDATION_PROMPT
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)
"""

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # current directory
TRUNK_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))      # Move up one level to reach trunk
DATA_FILE = os.path.join(TRUNK_DIR, 'data', 'cleaned_merged_ieee_bank_10k_version2.csv')
LOG_FILE = os.path.join(TRUNK_DIR, 'logs', 'chatbot_logs.jsonl')
MANAGER_LOG_FILE = os.path.join(TRUNK_DIR, 'logs', 'manager_logs.jsonl')

def save_log_manager(event: dict):
    os.makedirs("logs", exist_ok=True)
    event["timestamp"] = datetime.now().isoformat()
    with open(MANAGER_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
        
def save_log(event: dict):
    os.makedirs("logs", exist_ok=True)
    event["timestamp"] = datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

# === Collaborative Filtering (SVD) Model ===
def collaborative_filtering_recommendations(df1):
    print("\n🤝 Collaborative Filtering Recommendations (SVD)")

    reader = Reader(rating_scale=(0, 1))
    data = Dataset.load_from_df(df1[['AccountID', 'Merchant', 'TransactionAmt']], reader)

    svd = SVD(n_factors=50, n_epochs=20, lr_all=0.005, reg_all=0.02, random_state=42)
    cross_validate(svd, data, measures=['RMSE', 'MAE'], cv=3, verbose=True)
    trainset = data.build_full_trainset()
    svd.fit(trainset)

    return svd

#RiskProfile, CreditScore,LoanEligibilityScore, CustomerTenure
# === Content-Based Filtering (KNN) Model ===
def content_based_recommendations(df1):
    print("\n🤝 Content-Based Recommendations (KNN)")

    customer_profiles = df1.groupby('AccountID').agg({
        'TransactionAmt': ['mean', 'count', 'sum'],
        'CreditScore': 'mean',
        'CustomerAge': 'mean',
        'CustomerOccupation': 'first',
        'PreferredSpendingCategory': 'first',
        'MostInterestedProduct': 'first'
    }).reset_index()

    customer_profiles.columns = [
        'AccountID', 'AvgAmount', 'TransactionCount', 'TotalAmount',
        'CreditScore', 'Age', 'Occupation', 'PreferredSpendingCategory', 'MostInterestedProduct'
    ]

    customer_profiles_encoded = pd.get_dummies(
        customer_profiles, 
        columns=['Occupation', 'PreferredSpendingCategory']
    )

    scaler = StandardScaler()
    feature_cols = customer_profiles_encoded.columns.difference(['AccountID', 'MostInterestedProduct'])
    customer_profiles_encoded[feature_cols] = scaler.fit_transform(customer_profiles_encoded[feature_cols])

    knn = NearestNeighbors(n_neighbors=5, metric='cosine')
    knn.fit(customer_profiles_encoded[feature_cols])

    return knn, customer_profiles, customer_profiles_encoded, feature_cols


# === Combined Recommendation Output ===
def generate_customer_recommendation(customer_id, df1, knn_model, customer_profiles, customer_profiles_encoded, feature_cols):
    # Content-based output
    try:
        most_interested_product = customer_profiles[customer_profiles['AccountID'] == customer_id]['MostInterestedProduct'].values[0]
    except IndexError:
        return "Customer ID not found.", "", ""

    # Find similar customers using KNN
    idx = customer_profiles_encoded[customer_profiles_encoded['AccountID'] == customer_id].index
    if len(idx) == 0:
        return "Customer not found in encoded profiles.", "", ""
    
    customer_vector = customer_profiles_encoded.loc[idx[0], feature_cols].values.reshape(1, -1)
    distances, indices = knn_model.kneighbors(customer_vector, n_neighbors=3)

    similar_indices = indices[0][1:]  # Exclude self
    similar_customers = customer_profiles_encoded.iloc[similar_indices]['AccountID'].values

    similar_products = []
    for cust_id in similar_customers:
        product = customer_profiles[customer_profiles['AccountID'] == cust_id]['MostInterestedProduct'].values[0]
        similar_products.append(product)

    return most_interested_product, similar_products[0], similar_products[1]

# === Run Everything ===
def run_recommendation_pipeline(df1, test_account_id):
    print(f"\n🧪 Generating recommendations for AccountID = {test_account_id}")

    svd_model = collaborative_filtering_recommendations(df1)
    knn_model, customer_profiles, customer_profiles_encoded, feature_cols = content_based_recommendations(df1)

    output1, output2, output3 = generate_customer_recommendation(
        test_account_id, df1, knn_model, customer_profiles, customer_profiles_encoded, feature_cols
    )

    print("\n🎯 Final Recommendation Output:")
    print(f"1. Customer's own most interested product: {output1}")
    print(f"2. Similar customer #1's most interested product: {output2}")
    print(f"3. Similar customer #2's most interested product: {output3}")
   
    return output1, output2, output3
    
def render(user_id, account_id):

    try:
        
        df1 = pd.read_csv(DATA_FILE)
        # user_df = df[df["AccountID"] == account_id]
        # transactions = user_df.to_string(index=False)

        product1,product2, product3 = run_recommendation_pipeline(df1,account_id) 

        st.markdown("**Based on your browsing history:**")
        st.write("      \" " + product1 + "\"")

        st.write("\n\n\n\n\n\n\n\n\n\n\n")

        st.markdown("**Based on customers similar to you:**")
        st.write("      \" " + product2 + "\"")
        st.write("      \" " + product3 + "\"")

        save_log({
            "user_id": user_id,
            "account_id": account_id,
            "agent_called": "generate_recommendations",
            "action": "Recommendations Generated",
            "user_query": "Static Recommendations Generated",
            "result_summary": f"Products: {product1}, {product2}, {product3}"
        })

        save_log_manager({
            "user_id": user_id,
            "account_id": account_id,
            "agent_called": "Recommendations",
            "action": "Recommendations Generated",
            "result_summary": f"Products: {product1}, {product2}, {product3}"
        })

    except Exception as e:

        error_message = "Apologies, something went wrong. Please try again later."
        st.markdown(error_message)

        save_log({
            "user_id": user_id,
            "account_id": account_id,
            "agent_called": "generate_recommendations",
            "action": "Error generating Recommendations",
            "user_query": "Generate Recommendations",
            "result_summary": str(e)
        })

        save_log_manager({
            "user_id": user_id,
            "account_id": account_id,
            "agent_called": "Recommendations",
            "action": "Error generating Recommendations",
            "result_summary": str(e)
        })
