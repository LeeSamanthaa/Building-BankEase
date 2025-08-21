import pandas as pd
import numpy as np
import joblib
import os

from sklearn.metrics.pairwise import cosine_similarity

# Load models with error handling
try:
    scaler_profile = joblib.load("models/customer_profile_scaler.pkl")
    print("Scaler loaded successfully")
except Exception as e:
    print(f"Error loading scaler: {e}")
    scaler_profile = None

try:
    kmeans_model = joblib.load("models/kmeans_cluster_model.pkl")
    print("KMeans model loaded successfully")
except Exception as e:
    print(f"Error loading KMeans model: {e}")
    kmeans_model = None

try:
    label_encoders = {
        "TransactionType": joblib.load("models/label_encoder_TransactionType.pkl"),
        "Location": joblib.load("models/label_encoder_Location.pkl"),
        "Channel": joblib.load("models/label_encoder_Channel.pkl"),
        "CustomerOccupation": joblib.load("models/label_encoder_CustomerOccupation.pkl"),
    }
    print("Label encoders loaded successfully")
except Exception as e:
    print(f"Error loading label encoders: {e}")
    label_encoders = {}

try:
    svd_model = joblib.load("models/svd_model.pkl")
    print("SVD model loaded successfully")
except Exception as e:
    print(f"Error loading SVD model: {e}")
    svd_model = None


def load_data():
    try:
        if not os.path.exists("data/bank_transactions_data_2.csv"):
            raise FileNotFoundError("Data file not found: data/bank_transactions_data_2.csv")

        df = pd.read_csv("data/bank_transactions_data_2.csv", parse_dates=["TransactionDate"])
        print(f"Data loaded successfully. Shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        raise


def preprocess_customer_profiles(df):
    customer_profiles = df.groupby("AccountID").agg({
        'TransactionAmount': ['count', 'sum', 'mean'],
        'CustomerAge': 'first',
        'TransactionDuration': 'mean',
        'LoginAttempts': 'mean',
        'AccountBalance': 'mean'
    }).reset_index()

    customer_profiles.columns = [
        'AccountID', 'Total_Transactions', 'Total_Spending', 'Avg_Transaction_Amount',
        'Age', 'Avg_Transaction_Duration', 'Avg_Login_Attempts', 'Avg_Account_Balance'
    ]

    customer_profiles.fillna(0, inplace=True)
    return customer_profiles


def load_customer_profiles(df):
    """Load and preprocess customer profiles from the dataframe"""
    try:
        print("Creating customer profiles...")

        # Check if required columns exist
        required_cols = ['AccountID', 'TransactionAmount', 'CustomerAge', 'TransactionDuration',
                         'LoginAttempts', 'AccountBalance', 'CustomerOccupation']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            print(f"Missing columns: {missing_cols}")
            print(f"Available columns: {list(df.columns)}")

        customer_profiles = df.groupby("AccountID").agg({
            'TransactionAmount': ['count', 'sum', 'mean'],
            'CustomerAge': 'first',
            'TransactionDuration': 'mean',
            'LoginAttempts': 'mean',
            'AccountBalance': 'mean',
            'CustomerOccupation': 'first'  # Added occupation
        }).reset_index()

        # Try multiple possible column name combinations
        customer_profiles.columns = [
            'AccountID', 'Total_Transactions', 'Total_Spending', 'Avg_Transaction_Amount',
            'Age', 'Avg_Transaction_Duration', 'Avg_Login_Attempts', 'Avg_Account_Balance',
            'Occupation'
        ]

        customer_profiles.fillna(0, inplace=True)
        customer_profiles.set_index('AccountID', inplace=True)

        print(f"Customer profiles created successfully. Shape: {customer_profiles.shape}")
        print(f"Profile columns: {list(customer_profiles.columns)}")

        return customer_profiles
    except Exception as e:
        print(f"Error creating customer profiles: {e}")
        raise


def build_similarity_matrix(profiles):
    """Build similarity matrix from customer profiles"""
    try:
        # Debug: Print available columns and scaler info
        print(f"Available profile columns: {list(profiles.columns)}")

        # Check if scaler has feature_names_in_ attribute (sklearn >= 1.0)
        if hasattr(scaler_profile, 'feature_names_in_'):
            print(f"Scaler expects features: {list(scaler_profile.feature_names_in_)}")
            expected_features = list(scaler_profile.feature_names_in_)
        else:
            print("Scaler doesn't have feature_names_in_ attribute")
            # Try with the most common feature combinations
            expected_features = ['Total_Transactions', 'Total_Spending', 'Avg_Transaction_Amount',
                                 'Age', 'Avg_Transaction_Duration', 'Avg_Login_Attempts']

        # Check which features are actually available
        available_features = [col for col in expected_features if col in profiles.columns]
        print(f"Available expected features: {available_features}")

        if len(available_features) < 4:
            raise ValueError(f"Not enough matching features found. Available: {available_features}")

        # Use the available features
        profile_matrix = scaler_profile.transform(profiles[available_features])
        similarity_matrix = cosine_similarity(profile_matrix)
        return similarity_matrix
    except Exception as e:
        print(f"Error in build_similarity_matrix: {e}")
        raise


def get_recommendations(customer_id, similarity_matrix, profiles, top_k=3):
    """Get recommendations for a specific customer"""
    if customer_id not in profiles.index:
        return []

    customer_idx = profiles.index.get_loc(customer_id)
    similarities = similarity_matrix[customer_idx]

    # Get indices of most similar customers (excluding the customer itself)
    similar_indices = np.argsort(similarities)[-top_k - 1:-1][::-1]

    # Get the account IDs of similar customers
    similar_customers = profiles.iloc[similar_indices].index.tolist()
    return similar_customers


def recommend_similar_customers(customer_profiles, top_k=3):
    # Use the exact feature names that the scaler expects
    feature_cols = ['Total_Transactions', 'Total_Spending', 'Avg_Transaction_Amount',
                    'Age', 'TransactionDuration', 'LoginAttempts']
    profile_matrix = scaler_profile.transform(customer_profiles[feature_cols])
    similarity = cosine_similarity(profile_matrix)

    recommendations = {}

    for i, account_id in enumerate(customer_profiles['AccountID']):
        sims = similarity[i]
        sim_indices = np.argsort(sims)[-top_k - 1:-1][::-1]
        similar_accounts = customer_profiles.iloc[sim_indices]['AccountID'].tolist()
        recommendations[account_id] = similar_accounts

    return recommendations


def get_channel_recommendation(profile_row, cluster_model):
    product_mapping = {
        'Branch': 'Personal Banker, Cybersecurity Assessment, Credit Cards',
        'ATM': 'PIN Change, Balance Inquiry',
        'Online': 'Online Coupons',
    }

    try:
        # Debug: Check what features the scaler expects
        if hasattr(scaler_profile, 'feature_names_in_'):
            expected_features = list(scaler_profile.feature_names_in_)
            print(f"Scaler expects features: {expected_features}")
        else:
            # Fallback to common feature names
            expected_features = ['Total_Transactions', 'Total_Spending', 'Avg_Transaction_Amount',
                                 'Age', 'Avg_Transaction_Duration', 'Avg_Login_Attempts']

        # Check which features are available in the profile
        available_features = [col for col in expected_features if col in profile_row.index]
        print(f"Available features in profile: {available_features}")

        if len(available_features) < 4:
            raise ValueError(f"Not enough matching features. Available: {available_features}")

        # Use the available features
        feature_vector = profile_row[available_features].values.reshape(1, -1)
        scaled_vector = scaler_profile.transform(feature_vector)
        cluster = cluster_model.predict(scaled_vector)[0]

        cluster_to_channel = {
            0: 'Branch',
            1: 'ATM',
            2: 'Online',
            3: 'Branch'
        }

        channel = cluster_to_channel.get(cluster, 'Branch')
        return channel, product_mapping[channel]
    except Exception as e:
        print(f"Error in get_channel_recommendation: {e}")
        # Return default values if there's an error
        return 'Branch', product_mapping['Branch']