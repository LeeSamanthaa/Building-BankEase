import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity


def preprocess_customer_profiles(df):
    df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])
    customer_profiles = df.groupby('AccountID').agg({
        'TransactionAmount': ['count', 'sum', 'mean', 'std'],
        'CustomerAge': 'first',
        'CustomerOccupation': 'first',
        'TransactionDuration': 'mean',
        'LoginAttempts': ['mean', 'max'],
        'AccountBalance': 'mean',
        'TransactionDate': ['min', 'max']
    }).round(2)

    customer_profiles.columns = ['_'.join(col).strip() for col in customer_profiles.columns]
    column_mapping = {
        'TransactionAmount_count': 'Total_Transactions',
        'TransactionAmount_sum': 'Total_Spending',
        'TransactionAmount_mean': 'Avg_Transaction_Amount',
        'TransactionAmount_std': 'Transaction_Amount_Std',
        'CustomerAge_first': 'Age',
        'CustomerOccupation_first': 'Occupation',
        'TransactionDuration_mean': 'Avg_Transaction_Duration',
        'LoginAttempts_mean': 'Avg_Login_Attempts',
        'LoginAttempts_max': 'Max_Login_Attempts',
        'AccountBalance_mean': 'Avg_Account_Balance',
        'TransactionDate_min': 'First_Transaction_Date',
        'TransactionDate_max': 'Last_Transaction_Date'
    }
    customer_profiles = customer_profiles.rename(columns=column_mapping)

    customer_profiles['Customer_Tenure_Days'] = (
            customer_profiles['Last_Transaction_Date'] - customer_profiles['First_Transaction_Date']
    ).dt.days

    reference_date = df['TransactionDate'].max()
    customer_profiles['Days_Since_Last_Transaction'] = (
            reference_date - customer_profiles['Last_Transaction_Date']
    ).dt.days

    return customer_profiles


def compute_similarity_and_recommendations(customer_profiles, k=3):
    features = ['Total_Transactions', 'Total_Spending', 'Avg_Transaction_Amount', 'Age']
    profile_features = customer_profiles[features].fillna(0)
    scaler = StandardScaler()
    normalized_profiles = scaler.fit_transform(profile_features)

    similarity_matrix = cosine_similarity(normalized_profiles)

    top_k_recommendations = {}
    for idx, account_id in enumerate(customer_profiles.index):
        top_indices = similarity_matrix[idx].argsort()[-(k + 1):-1][::-1]
        top_k_recommendations[account_id] = customer_profiles.index[top_indices].tolist()

    return similarity_matrix, top_k_recommendations
