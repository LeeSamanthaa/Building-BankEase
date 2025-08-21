# manager_fraud_feedback_view.py

import streamlit as st
import os
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
TRUNK_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
FLAG_FILE = os.path.join(TRUNK_DIR, 'flags', 'flag_fraud.csv')

def render(user_id):
    if not os.path.exists(FLAG_FILE):
        st.warning("No fraud feedback data available.")
        return

    try:
        df = pd.read_csv(FLAG_FILE)
    except Exception as e:
        st.error(f"Error reading fraud feedback file: {e}")
        return

    required_columns = [
        "guid", "TransactionDate", "TransactionAmt", "Merchant", "timestamp",
        "probability", "alert", "manager_decision", "notes"
    ]
    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        st.warning(f"Missing required columns in CSV: {', '.join(missing)}")
        return

    user_feedback = df[df["UserID"] == user_id]
    if user_feedback.empty:
        st.info("No fraud feedback records found for your account.")
        return

    # 🔔 Show alert if any transaction is flagged
    if (user_feedback["manager_decision"] == "Flag as Fraud").any():
        st.error("⚠️ ALERT: Your recent transaction was flagged as **FRAUD** by the manager!")

    st.subheader("Manager feedback (select transactions)")
    st.dataframe(
        user_feedback[required_columns],
        use_container_width=True
    )
