import streamlit as st
import pandas as pd
import joblib
import os
from datetime import datetime
import numpy as np

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRUNK_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))

MODEL_PATH = os.path.join(TRUNK_DIR, 'models', 'fraud_detection_artifacts.pkl')
DATA_PATH = os.path.join(TRUNK_DIR, 'data', 'cleaned_merged_ieee_bank_10k_version2.csv')
PENDING_PATH = os.path.join(TRUNK_DIR, 'flags', 'flag_fraud_pending.csv')
REVIEW_PATH = os.path.join(TRUNK_DIR, 'flags', 'flag_fraud.csv')

artifacts = joblib.load(MODEL_PATH)
models = artifacts['models']
feature_names = artifacts['feature_names']
imputation_values = artifacts['imputation_values']
label_encoders = artifacts['label_encoders']
chatbot_config = artifacts['chatbot_config']
uid_cols = artifacts['uid_cols']
DATE_COLUMN = 'TransactionDate'

def create_leak_free_features(df, uid_cols, date_col):
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(by=date_col).reset_index(drop=True)
    user_col = next((col for col in uid_cols if col in df.columns), None)
    if not user_col: return df

    df[f'{user_col}_count_expanding'] = df.groupby(user_col).cumcount()
    df[f'{user_col}_amount_mean_expanding'] = df.groupby(user_col)['TransactionAmt'].expanding().mean().reset_index(level=0, drop=True)
    for window in ['30min', '1h', '24h', '7D']:
        grouped = df.groupby(user_col).rolling(window, on=date_col)['TransactionAmt']
        for agg in ['mean', 'sum', 'std', 'min', 'max']:
            df[f'{user_col}_amount_{agg}_{window}'] = grouped.agg(agg).values
        df[f'{user_col}_txn_count_{window}'] = grouped.count().values
    if 'ProductCD' in df.columns:
        df['time_since_last_product_txn'] = df.groupby([user_col, 'ProductCD'])[date_col].diff().dt.total_seconds().fillna(0)
    if 'Merchant' in df.columns:
        df['time_since_last_merchant_txn'] = df.groupby([user_col, 'Merchant'])[date_col].diff().dt.total_seconds().fillna(0)
    for col in df.columns:
        if any(x in col for x in [f'{user_col}_', 'time_since']):
            df[col] = df.groupby(user_col)[col].shift(1)
    return df

def predict_fraud_ensemble(models, df):
    preds = np.array([model.predict_proba(df)[:, 1] for model in models])
    return preds.mean(axis=0)

def generate_chatbot_response(prob, config):
    if prob >= config['confidence_levels']['very_high']:
        return config['alert_messages']['very_high']
    elif prob >= config['confidence_levels']['high']:
        return config['alert_messages']['high']
    elif prob >= config['low_risk_threshold']:
        return config['alert_messages']['medium']
    else:
        return config['alert_messages']['low']

def render():
    st.subheader("🛡️ Manager Fraud Review Panel")

    if not os.path.exists(PENDING_PATH):
        st.info("No pending fraud transactions.")
        return

    df_pending = pd.read_csv(PENDING_PATH)
    if df_pending.empty:
        st.info("No transactions pending review.")
        return

    flagged_indices = []
    probabilities = []

    for idx, row in df_pending.iterrows():
        user_id = row['UserID']
        df_history = pd.read_csv(DATA_PATH)
        user_history = df_history[df_history['UserID'] == user_id]
        combined = pd.concat([user_history, pd.DataFrame([row])], ignore_index=True)
        processed = create_leak_free_features(combined.copy(), uid_cols, DATE_COLUMN)

        for col, val in imputation_values.items():
            if col in processed.columns:
                processed[col] = processed[col].fillna(val['value'])
        for col, le in label_encoders.items():
            if col in processed.columns:
                processed[col] = processed[col].astype(str).apply(lambda x: x if x in le.classes_ else 'UNKNOWN')
                processed[col] = le.transform(processed[col])

        final_txn_df = processed.iloc[-1:].reindex(columns=feature_names, fill_value=0)
        prob = predict_fraud_ensemble(models, final_txn_df)[0]

        if row['TransactionType'].lower() == 'debit' and (prob >= chatbot_config['confidence_levels']['high'] or row['TransactionAmt'] >= 2000):
            flagged_indices.append(idx)
            probabilities.append(prob)

    if not flagged_indices:
        st.success("✅ No high-risk transactions to review.")
        return

    selected_index = st.selectbox("Select a flagged transaction to review", flagged_indices)
    selected_txn = df_pending.iloc[selected_index].to_dict()
    prob = probabilities[flagged_indices.index(selected_index)]
    chatbot_msg = generate_chatbot_response(prob, chatbot_config)

    st.markdown(f"**🧠 Model Fraud Probability:** `{prob:.2%}`")
    st.markdown(f"**🤖 Alert:** {chatbot_msg}")
    st.dataframe(pd.DataFrame([selected_txn]).iloc[:, 1:])

    decision = st.selectbox("Manager Decision", ["Flag as not fraud", "Flag as Fraud"])
    notes = st.text_area("Manager Notes")

    if st.button("✅ Submit Review"):

        if prob >= chatbot_config['confidence_levels']['high'] and decision == "Flag as not fraud":
            st.warning("⚠️ Model predicts FRAUD. Are you sure you want to APPROVE this transaction as safe?")

        log = {
            **selected_txn,
            "probability": float(prob),
            "alert": chatbot_msg,
            "manager_decision": decision,
            "notes": notes,
            "timestamp": datetime.now().isoformat(),
            "UserID": selected_txn["UserID"]
        }
       
        os.makedirs(os.path.dirname(REVIEW_PATH), exist_ok=True)
        pd.DataFrame([log]).to_csv(REVIEW_PATH, mode='a', header=not os.path.exists(REVIEW_PATH), index=False)

        st.success("✅ Review submitted.")

