import streamlit as st
import pandas as pd
import joblib
import traceback
from customer_utils import (
    load_data, load_customer_profiles,
    build_similarity_matrix, get_recommendations, get_channel_recommendation
)

# Load Data and Models
st.set_page_config("Customer Insights", layout="wide")

# Add error handling and debug information
try:
    # st.write("Loading data...")
    df = load_data()
    #st.write(f"Data loaded successfully! Shape: {df.shape}")

    #st.write("Processing customer profiles...")
    profiles = load_customer_profiles(df)
    #st.write(f"Profiles created successfully! Shape: {profiles.shape}")
    st.write(f"Profile columns: {list(profiles.columns)}")

    #st.write("Building similarity matrix...")
    similarity_matrix = build_similarity_matrix(profiles)
    #st.write(f"Similarity matrix built successfully! Shape: {similarity_matrix.shape}")

    #st.write("Loading cluster model...")
    cluster_model = joblib.load("models/kmeans_cluster_model.pkl")
    st.write("Cluster model loaded successfully!")

except Exception as e:
    st.error(f"Error during initialization: {str(e)}")
    st.error(f"Traceback: {traceback.format_exc()}")
    st.stop()

# Sidebar
st.sidebar.title("📊 Navigation")
tab = st.sidebar.radio("Go to", ["Customer Statistics", "Customer Profile Explorer & Recommender"])

# -------- TAB 1: Customer Statistics --------
if tab == "Customer Statistics":
    st.title("📈 Customer Statistics")
    try:
        st.markdown("### Overview")
        st.metric("Total Customers", f"{df['AccountID'].nunique():,}")
        st.metric("Total Transactions", f"{len(df):,}")
        st.metric("Average Transaction Amount", f"${df['TransactionAmount'].mean():.2f}")

        st.markdown("### Transaction Type Distribution")
        st.bar_chart(df['TransactionType'].value_counts())

        st.markdown("### Channel Usage")
        st.bar_chart(df['Channel'].value_counts())
    except Exception as e:
        st.error(f"Error in Customer Statistics: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")

# -------- TAB 2: Profile Explorer & Recommender --------
elif tab == "Customer Profile Explorer & Recommender":
    st.title("👤 Customer Profile Recommender")
    try:
        st.markdown("This section displays **recommended customers** per transaction channel (Branch, ATM, Online).")

        # Segment customers by their dominant channel
        channel_categories = ['Branch', 'ATM', 'Online']

        for channel in channel_categories:
            st.subheader(f"📌 Channel: {channel}")
            st.markdown("---")

            # Filter random 3 profiles
            sample_profiles = profiles.sample(3, random_state=42)

            cols = st.columns(3)
            for i, rec_id in enumerate(sample_profiles.index):
                profile = sample_profiles.loc[rec_id]
                channel_rec, product_text = get_channel_recommendation(profile, cluster_model)

                with cols[i]:
                    st.markdown(f"**Customer ID:** `{rec_id}`")
                    st.markdown(f"- Age: {profile['Age']}")
                    st.markdown(f"- Occupation: {profile['Occupation']}")
                    st.markdown(f"- Total Spending: ${profile['Total_Spending']:,.2f}")
                    st.markdown(f"- 📌 Recommended Channel: **{channel_rec}**")
                    st.markdown(f"🛍️ **Products:** {product_text}")
                    st.button("Send Offer", key=f"btn_{rec_id}_{i}_{channel}")  # Added channel to make key unique
    except Exception as e:
        st.error(f"Error in Profile Explorer: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")