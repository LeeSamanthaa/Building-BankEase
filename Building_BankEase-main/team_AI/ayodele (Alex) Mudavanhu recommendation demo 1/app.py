import streamlit as st
import pandas as pd
from customer_utils import preprocess_customer_profiles, compute_similarity_and_recommendations

# Load dataset
df = pd.read_csv("bank_transactions_data_2.csv")
df['TransactionDate'] = pd.to_datetime(df['TransactionDate'])

# Build customer profiles and recommendations
customer_profiles = preprocess_customer_profiles(df)
similarity_matrix, top_k_recommendations = compute_similarity_and_recommendations(customer_profiles)

# Streamlit UI
st.set_page_config(page_title="Banking App", layout="wide")
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Customer Statistics", "Customer Profile Explorer & Recommender"])

if page == "Customer Statistics":
    st.title("📊 Customer Statistics Overview")
    st.dataframe(customer_profiles.describe())

elif page == "Customer Profile Explorer & Recommender":
    st.title("🧠 Customer Profile Explorer & Recommender")
    customer_id = st.selectbox("Select a Customer", customer_profiles.index.astype(str))

    if customer_id:
        st.subheader(f"Profile for {customer_id}")
        st.dataframe(customer_profiles.loc[[customer_id]])

        st.markdown("---")
        st.subheader("🧭 Recommended Similar Customers")
        rec_ids = top_k_recommendations.get(customer_id, [])

        cols = st.columns(len(rec_ids))
        for i, rec_id in enumerate(rec_ids):
            with cols[i]:
                st.markdown(f"**Customer ID:** {rec_id}")
                st.dataframe(customer_profiles.loc[[rec_id]])
