import joblib
import pandas as pd
import numpy as np

# --- 1. Load the deployment artifacts ---
MODEL_PATH = 'fraud_detection_artifacts.pkl'
DATA_PATH = 'cleaned_merged_ieee_bank_10k_version2.csv'

print("Loading deployment artifacts...")
try:
    artifacts = joblib.load(MODEL_PATH)
    print("Artifacts loaded successfully.")
except FileNotFoundError:
    print(f"ERROR: Deployment file '{MODEL_PATH}' not found. Please ensure the training pipeline has been run.")
    exit()
except ModuleNotFoundError as e:
    print(f"ERROR: Required module not found: {e}. Please ensure all necessary libraries are installed.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading artifacts: {e}")
    exit()

models = artifacts['models']
feature_names = artifacts['feature_names']
imputation_values = artifacts['imputation_values']
label_encoders = artifacts['label_encoders']
chatbot_config = artifacts['chatbot_config']
uid_cols = artifacts['uid_cols']

# Determine the correct date column from the feature names
DATE_COLUMN = 'TransactionDate'
if 'TransactionDT' in feature_names:
    DATE_COLUMN = 'TransactionDT'

# --- 2. Define the feature engineering function (replicated from training) ---
def create_leak_free_features(df, uid_cols, DATE_COLUMN):
    """
    Recreates the leak-free features for a given DataFrame.
    This function must exactly match the feature engineering in the training pipeline.
    """
    if not uid_cols or 'TransactionAmt' not in df.columns:
        print("Warning: User ID or Transaction Amount column not found. Skipping aggregation features.")
        return df

    # Ensure date column is datetime and sort
    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors='coerce')
    df = df.dropna(subset=[DATE_COLUMN]).sort_values(by=DATE_COLUMN).reset_index(drop=True)

    # Use the first valid UID column for feature creation
    user_col = None
    for col in uid_cols:
        if col in df.columns:
            user_col = col
            break

    if user_col is None:
        print("Warning: No valid user ID column found for feature creation. Skipping aggregation features.")
        return df

    # Expanding Features
    df[f'{user_col}_count_expanding'] = df.groupby(user_col).cumcount()
    df[f'{user_col}_amount_mean_expanding'] = df.groupby(user_col)['TransactionAmt'].expanding().mean().reset_index(level=0, drop=True)

    # Rolling Features (using 'h' for hours as per FutureWarning fix)
    rolling_windows = ['30min', '1h', '24h', '7D']
    rolling_aggs = ['mean', 'sum', 'std', 'min', 'max']
    for window in rolling_windows:
        rolling_features = df.groupby(user_col).rolling(window, on=DATE_COLUMN)['TransactionAmt'].agg(rolling_aggs)
        for agg in rolling_aggs:
            df[f'{user_col}_amount_{agg}_{window}'] = rolling_features[agg].values
        rolling_count = df.groupby(user_col).rolling(window, on=DATE_COLUMN)['TransactionAmt'].count()
        df[f'{user_col}_txn_count_{window}'] = rolling_count.values

    # Time Since Last Event Features
    if 'ProductCD' in df.columns:
        df['time_since_last_product_txn'] = df.groupby([user_col, 'ProductCD'])[DATE_COLUMN].diff().dt.total_seconds().fillna(0)
    if 'Merchant' in df.columns:
        df['time_since_last_merchant_txn'] = df.groupby([user_col, 'Merchant'])[DATE_COLUMN].diff().dt.total_seconds().fillna(0)

    # Shift all new features by 1 to ensure they are leak-free (only use past info)
    agg_cols = [col for col in df.columns if f'{user_col}_' in col or 'time_since' in col]
    for col in agg_cols:
        df[col] = df.groupby(user_col)[col].shift(1)

    return df

# --- 3. Define the prediction function ---
def predict_fraud_ensemble(models, df_processed):
    """
    Makes a fraud probability prediction using an ensemble of models.
    """
    all_model_preds = np.array([model.predict_proba(df_processed)[:, 1] for model in models])
    mean_prob = all_model_preds.mean(axis=0)
    return mean_prob

# --- 4. Define the chatbot response generation function ---
def generate_chatbot_response(probability, config):
    """Generates a human-readable chatbot message based on fraud probability."""
    if probability >= config['confidence_levels']['very_high']:
        return config['alert_messages']['very_high']
    elif probability >= config['confidence_levels']['high']:
        return config['alert_messages']['high']
    elif probability >= config['low_risk_threshold']:
        return config['alert_messages']['medium']
    else:
        return config['alert_messages']['low']

# --- MAIN SIMULATION BLOCK ---
if __name__ == "__main__":
    print("\n--- Simulating a New Transaction for Chatbot Response ---")

    new_transaction_data = {
        'TransactionAmt': 1500.00,
        'UserID': 'U005',
        'TransactionDate': '2025-07-20 14:30:00',
        'ProductCD': 'W',
        'Merchant': 'ElectroWorld',
        'isFraud': 0
    }

    # -- Doesnt use simulation uses our previous dataset for historical transaction --
    print("Loading historical data...")
    try:
        historical_data_for_user = pd.read_csv(DATA_PATH)
    except FileNotFoundError:
        print(f"ERROR: Historical data file '{DATA_PATH}' not found. Please check the filename and path.")
        exit()
    except KeyError as e:
        print(f"ERROR: A required column was not found in the CSV file: {e}. Please check your column names.")
        exit()

    # Filter the data for the specific user you are interested in
    historical_data_for_user = historical_data_for_user[historical_data_for_user['UserID'] == new_transaction_data['UserID']]
    print("Historical data loaded and filtered successfully.")
    # --- END OF SECTION TO REPLACE ---

    # Combine new transaction with historical data for feature engineering
    combined_data = pd.concat([historical_data_for_user, pd.DataFrame([new_transaction_data])], ignore_index=True)

    # Apply feature engineering
    df_processed_with_features = create_leak_free_features(combined_data.copy(), uid_cols, DATE_COLUMN)

    # Impute missing values (if any)
    for col, data in imputation_values.items():
        if col in df_processed_with_features.columns:
            df_processed_with_features[col] = df_processed_with_features[col].fillna(data['value'])

    # Apply label encoding
    for col, le in label_encoders.items():
        if col in df_processed_with_features.columns:
            df_processed_with_features[col] = df_processed_with_features[col].astype(str).apply(lambda x: x if x in le.classes_ else 'UNKNOWN')
            df_processed_with_features[col] = le.transform(df_processed_with_features[col])

    # Align the DataFrame with the model's expected feature names
    final_transaction_df = df_processed_with_features.iloc[-1:].reindex(columns=feature_names, fill_value=0)

    # Get the fraud probability
    fraud_probability = predict_fraud_ensemble(models, final_transaction_df)

    # Generate the chatbot response
    chatbot_message = generate_chatbot_response(fraud_probability[0], chatbot_config)

    print(f"\nTransaction Details:")
    print(f"  User ID: {new_transaction_data['UserID']}")
    print(f"  Amount: ${new_transaction_data['TransactionAmt']:.2f}")
    print(f"  Date: {new_transaction_data['TransactionDate']}")
    print(f"  Predicted Fraud Probability: {fraud_probability[0]:.2%}")
    print(f"  Chatbot Response: {chatbot_message}")
