import streamlit as st
import os
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import numpy as np
# Set paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRUNK_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
LOG_FILE = os.path.join(TRUNK_DIR, 'logs', 'chatbot_logs.jsonl')
MANAGER_LOG_FILE = os.path.join(TRUNK_DIR, 'logs', 'manager_logs.jsonl')




TRUNK_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))  
MODEL_PATH = os.path.join(TRUNK_DIR, 'models', 'fraud_detection_artifacts_np.pkl')
DATA_PATH = os.path.join(TRUNK_DIR, 'data', 'cleaned_merged_ieee_bank_10k_version2.csv')


# Save logs
def save_log(event: dict):
    os.makedirs("logs", exist_ok=True)
    event["timestamp"] = datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

def save_log_manager(event: dict):
    os.makedirs("logs", exist_ok=True)
    event["timestamp"] = datetime.now().isoformat()
    with open(MANAGER_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")
###Defining functions for preprocessing
# 3. Function to create dynamic features
def create_leak_free_features(df, uid_cols, DATE_COLUMN):
    if not uid_cols or 'TransactionAmt' not in df.columns:
        st.success("Warning: Skipping aggregation features due to missing User ID or Transaction Amount.")
        return df

    df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN])
    df = df.sort_values(by=DATE_COLUMN)

    user_col = uid_cols[0]

    # Expanding Features
    df[f'{user_col}_count_expanding'] = df.groupby(user_col).cumcount()
    df[f'{user_col}_amount_mean_expanding'] = df.groupby(user_col)['TransactionAmt'].expanding().mean().reset_index(level=0, drop=True)

    # Rolling Features (Corrected 'H' to 'h')
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

    # Shift all new features to be leak-free
    agg_cols = [col for col in df.columns if f'{user_col}_' in col or 'time_since' in col]
    for col in agg_cols:
        df[col] = df.groupby(user_col)[col].shift(1)

    return df

# 4. Function to predict
def predict_fraud_ensemble(models, df_processed, threshold):
    all_model_preds = np.array([model.predict_proba(df_processed)[:, 1] for model in models])
    mean_prob = all_model_preds.mean(axis=0)
    predictions = (mean_prob >= threshold).astype(int)
    return mean_prob, predictions

# Main render function
def render(user_id, account_id):
    st.subheader("Fraud Transactions Analysis")
    # 2. Load the model and artifacts
    try:
        
        # Leer el archivo
        with open(MODEL_PATH, 'rb') as f:
            artifacts = pickle.load(f)

        models = artifacts['models']
        feature_names = artifacts['feature_names']
        imputation_values = artifacts['imputation_values']
        label_encoders = artifacts['label_encoders']
        chatbot_config = artifacts['chatbot_config']
        uid_cols = artifacts['uid_cols']
        DATE_COLUMN = 'TransactionDate'

        if 'TransactionDT' in feature_names:
            DATE_COLUMN = 'TransactionDT'

        st.success("Artifacts loaded successfully.")
    except FileNotFoundError:
        st.success("ERROR: Deployment file not found. Please check your path.")
        exit()

    #Fraud Model Goes Here

    try:
        df = pd.read_csv(DATA_PATH, dtype={'guid': str})
        #df = pd.read_csv(DATA_PATH)
        ##Cleaning data

        ####

        # Check for required columns
        if "AccountID" not in df.columns or "isFraud" not in df.columns:
            st.error("Missing 'AccountID' or 'isFraud' column in the dataset.")
            return

        # Filter data for the user's account
        user_df = df[df["AccountID"] == account_id].copy()
        # Predict
        
        df_all_processed = create_leak_free_features(user_df, uid_cols, DATE_COLUMN)
            
        # 6. Impute values and encode
        for col, data in imputation_values.items():
            if col in df_all_processed.columns:
                df_all_processed[col] = df_all_processed[col].fillna(data['value'])
        # Preserve raw GUID before it gets label-encoded
        if 'guid' in df_all_processed.columns and 'guid_raw' not in df_all_processed.columns:
            df_all_processed['guid_raw'] = df_all_processed['guid'].astype(str).str.strip()

        for col, le in label_encoders.items():
            if col in df_all_processed.columns:
                df_all_processed[col] = df_all_processed[col].astype(str).apply(lambda x: x if x in le.classes_ else 'UNKNOWN')
                df_all_processed[col] = le.transform(df_all_processed[col])

        # 7. Align the DataFrame with the model's columns
        df_final = df_all_processed.reindex(columns=feature_names, fill_value=0)

        # 8. Make predictions and visualize the results
        threshold = chatbot_config['low_risk_threshold']
        probas, predictions = predict_fraud_ensemble(models, df_final, threshold=threshold)

        # Attach predictions back to the processed (but with preserved guid_raw)
        df_all_processed['FraudProbability'] = probas
        df_all_processed['Prediction'] = predictions

        # Build a small frame with just the join key + preds
        pred_cols = df_all_processed[['guid_raw', 'FraudProbability', 'Prediction']].copy()

        # Make sure raw user df has guid as string for a clean join
        user_df['guid_str'] = user_df['guid'].astype(str).str.strip()

        # Merge: raw user data (labels, channels, etc.) + predictions by GUID
        df_display_with_labels = pd.merge(
            user_df,
            pred_cols,
            left_on='guid_str',
            right_on='guid_raw',
            how='left'
        )

        # Sanity check for match coverage
        matched_ratio = df_display_with_labels['FraudProbability'].notna().mean()
        st.info(f"Matched predictions for {matched_ratio:.1%} of rows (by GUID).")

        # Keep only rows where model predicted fraud
        fraud_preds = df_display_with_labels[df_display_with_labels['Prediction'] == 1]

        if df_display_with_labels.empty:
            st.warning("No transactions found for this account.")
            return

        # Basic stats from merged display frame
        fraud_count = (df_display_with_labels['Prediction'] == 1).sum()
        total = len(df_display_with_labels)
        st.markdown(f"🔍 **Total Transactions:** {total}")
        st.markdown(f"🚨 **Fraudulent Transactions Detected:** {fraud_count}")

        # Human-friendly label column for plots
        df_display_with_labels['PredictionLabel'] = np.where(
            df_display_with_labels['Prediction'] == 1, 'Fraud',
            np.where(df_display_with_labels['Prediction'] == 0, 'Non-Fraud', 'No Prediction')
        )

        # Plot: Fraud vs Non-Fraud
        fig, ax = plt.subplots()
        sns.countplot(
            data=df_display_with_labels[df_display_with_labels['Prediction'].isin([0, 1])],
            x="PredictionLabel",
            palette="Set2",
            ax=ax
        )
        ax.set_title("Fraud vs Non-Fraud Transactions")
        st.pyplot(fig)

        # Fraud by Channel (if available)
        if 'Channel' in df_display_with_labels.columns:
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            sns.countplot(
                data=df_display_with_labels[df_display_with_labels['Prediction'].isin([0, 1])],
                x="Channel",
                hue="PredictionLabel",
                palette="Set1",
                ax=ax2
            )
            ax2.set_title("Fraud Distribution by Channel")
            st.pyplot(fig2)

        # Optional details table (only predicted frauds)
        if st.button("Show details of fraud transactions"):
            fraud_details = df_display_with_labels[df_display_with_labels['Prediction'] == 1][[
                'TransactionAmt', 'guid', 'TransactionDate', 'TransactionType', 'Merchant', 'Category', 'Channel'
            ]]
            st.markdown("### 📄 Fraud Transaction Details")
            st.dataframe(fraud_details)



        # Save logs
        summary = f"{fraud_count} frauds detected out of {total} transactions"
        save_log({
            "user_id": user_id,
            "account_id": account_id,
            "agent_called": "fraud_agent",
            "action": "Fraud Visualization",
            "user_query": "Static Fraud Detection (visual)",
            "result_summary": summary
        })

        save_log_manager({
            "user_id": user_id,
            "account_id": account_id,
            "agent_called": "Fraud",
            "action": "Fraud Visualization",
            "result_summary": summary
        })
   
    except Exception as e:

        st.error(f"Error: {e}")  # Add this line to see the real error
      
        save_log({
            "user_id": user_id,
            "account_id": account_id,
            "agent_called": "fraud_agent",
            "action": "Error during visualization",
            "user_query": "Fraud Detection",
            "result_summary": str(e)
        })
        save_log_manager({
            "user_id": user_id,
            "account_id": account_id,
            "agent_called": "Fraud",
            "action": "Error during visualization",
            "result_summary": str(e)
         })
