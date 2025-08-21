"""
Right now, running the script each time this agent is called
Since the response is static for a user, we can save the generated answer in session_state if customer returns back to this agent
This saves llm tokens
"""

import streamlit as st
import os
import pandas as pd
import json
from datetime import datetime
from agents.prompts import INSIGHTS_PROMPT
from langchain_groq import ChatGroq
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))          # current directory
TRUNK_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))      # Move up one level to reach trunk
DATA_FILE = os.path.join(TRUNK_DIR, 'data', 'cleaned_merged_ieee_bank_10k_chatbot_version2.csv')
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
        
# ----- UPDATED INSIGHTS VISUALISATION CODE -----


def render(user_id, account_id):

    try:
        df = pd.read_csv(DATA_FILE)
        df["TransactionDate"] = pd.to_datetime(df["TransactionDate"], errors='coerce')
        user_df = df[df["AccountID"] == account_id].copy()
       
        user_df["Year"] = user_df["TransactionDate"].dt.year
        user_df["Month"] = user_df["TransactionDate"].dt.month

        # --- Filter by year and month ---
        st.subheader("Filter by year and month")

        col1, col2 = st.columns(2)

        with col1:
            selected_year = st.selectbox("Select year", sorted(user_df["Year"].unique(), reverse=True))

        with col2:
            selected_month = st.selectbox("Select month", sorted(user_df["Month"].unique()))


        # --- Filered ---
        user_df = user_df[(user_df["Year"] == selected_year) & (user_df["Month"] == selected_month)]

        st.markdown(f"Showing results from **{selected_month:02d}/{selected_year}**")
        st.write(f"Found {len(user_df)} transactions.")

        # --- Graphics ---


        if user_df.empty:
            st.warning("No transaction data available for this account.")
            return

        # Split by type
        debit_df = user_df[user_df["TransactionType"] == "Debit"].copy()
        
        # --- Monthly Spending Bar Chart (Debits) ---
        st.markdown("### 📊 Monthly Spending Summary")

        if not debit_df.empty:
                debit_df.set_index("TransactionDate", inplace=True)

                # Monthly total
                total_spent = debit_df["TransactionAmt"].sum()
                month_label = debit_df.index[0].strftime("%B %Y")

                fig1, ax1 = plt.subplots(figsize=(6, 3))
                bar = ax1.bar([month_label], [total_spent], color="teal")

                # Add total spent label with padding above
                for rect in bar:
                        height = rect.get_height()
                        ax1.text(rect.get_x() + rect.get_width() / 2.0, height + 10, f"${total_spent:,.2f}",
                                 ha='center', va='bottom', fontsize=10)

                ax1.set_ylabel("Total Spent ($)")
                ax1.set_title(f"Total Debit Spend for {month_label}")
                ax1.set_ylim(0, total_spent * 1.2)  # Add space above bar
                st.pyplot(fig1)

                # --- Daily Spending Pattern ---
                st.markdown("### 📈 Daily Debit Spending Pattern")

                daily_spend = debit_df["TransactionAmt"].resample("D").sum()
                avg_value = debit_df["AvgMonthlySpend"].iloc[0]

                fig2, ax2 = plt.subplots(figsize=(8, 4))
                sns.barplot(x=daily_spend.index.strftime('%d %b'),y=daily_spend.values,ax=ax2,color="steelblue")  # Uniform single color
                ax2.axhline(avg_value, linestyle="--", color="red", label=f"Average Monthly Spending: ${avg_value:.2f}")
                ax2.set_ylabel("Amount Spent ($)")
                ax2.set_xlabel("Day")
                ax2.set_title(f"Daily Spending in {month_label}")
                ax2.tick_params(axis='x', rotation=45)
                ax2.legend()
                st.pyplot(fig2)

        # --- Transaction Type Pie Chart ---
        st.write("### 🔍 Transaction Type Distribution")
        type_counts = user_df["TransactionType"].value_counts()
        fig2, ax2 = plt.subplots()
        type_counts.plot(kind='pie', autopct='%1.1f%%', startangle=90, ax=ax2)
        ax2.set_ylabel("")
        ax2.set_title("Debit vs Credit Transaction Count")
        st.pyplot(fig2)

        # --- Category-wise Spending (Pie Chart) ---
        st.write("### 🧾 Spending by Category")
        if "Category" in debit_df.columns and not debit_df["Category"].isna().all():
            cat_spend = debit_df.groupby("Category")["TransactionAmt"].sum().sort_values(ascending=False)
            fig4, ax4 = plt.subplots()
            try:
                cat_spend.plot(kind="pie", autopct="%1.1f%%", ax=ax4)
                ax4.set_ylabel("")
                ax4.set_title("Spending Distribution by Category")
                st.pyplot(fig4)
            except Exception as e:
                st.error(f"Error generating pie chart: {e}")
        else:
            st.info("No category data available for pie chart.")

        # --- Top Merchants by Frequency ---
        st.write("### 🏪 Most Frequent Merchants")
        if "Merchant" in debit_df.columns:
            merchant_freq = debit_df["Merchant"].value_counts().head(5)
            fig5, ax5 = plt.subplots()
            sns.barplot(x=merchant_freq.values, y=merchant_freq.index, ax=ax5, palette="magma")
            ax5.set_title("Top 5 Frequent Merchants")
            ax5.set_xlabel("Transaction Count")
            st.pyplot(fig5)
            
      
        # --- Top Merchants by Total Debit Amount ---
        st.write("### 💸 High-Value Merchants (Total Spend)")
        if "Merchant" in debit_df.columns:
            merchant_amt = debit_df.groupby("Merchant")['TransactionAmt'].sum().sort_values(ascending=False).head(5)
            fig6, ax6 = plt.subplots()
            sns.barplot(x=merchant_amt.values, y=merchant_amt.index, ax=ax6, palette="coolwarm")
            ax6.set_title("Top 5 Merchants by Debit Spend")
            ax6.set_xlabel("Total Amount Spent ($)")
            st.pyplot(fig6)
            
        # --- Transaction Channel Distribution ---
        st.markdown("### 🛰️ Transaction Channels")

        if "Channel" in user_df.columns and not user_df["Channel"].isna().all():
                channel_counts = user_df["Channel"].value_counts()

                fig_channel, ax_channel = plt.subplots()
                sns.barplot(
                        x=channel_counts.values,
                        y=channel_counts.index,
                        palette="cubehelix",
                        ax=ax_channel
                )
                ax_channel.set_title("Transactions by Channel")
                ax_channel.set_xlabel("Transaction Count")
                ax_channel.set_ylabel("Channel")
                st.pyplot(fig_channel)
        else:
                st.info("No channel data available.")
        
        st.markdown("### 👤 Your Profile Summary")

        col1, col2 = st.columns(2)

        with col1:
                if "CustomerTenure" in user_df.columns:
                        tenure = int(user_df["CustomerTenure"].iloc[0])
                        st.markdown(f"**Customer Tenure**  \n<span style='font-size:24px'>{tenure} years</span>", unsafe_allow_html=True)

        with col2:
                if "CreditScore" in user_df.columns:
                        score = int(user_df["CreditScore"].iloc[0])
                        st.markdown(f"**Credit Score**  \n<span style='font-size:24px'>{score}</span>", unsafe_allow_html=True)                


        # --- Logging ---
        save_log({
            "user_id": user_id,
            "account_id": account_id,
            "agent_called": "generate_insights",
            "action": "Charts Only",
            "user_query": "Chart-based insights",
            "result_summary": "Charts rendered without LLM or forecast."
        })

        save_log_manager({
            "user_id": user_id,
            "account_id": account_id,
            "agent_called": "Insights",
            "action": "Charts Only",
            "result_summary": "Charts rendered without LLM or forecast."
        })

    except Exception as e:
        st.error("Error generating charts. Please try again.")

        save_log({
            "user_id": user_id,
            "account_id": account_id,
            "agent_called": "generate_insights",
            "action": "Chart Error",
            "user_query": "Chart-based insights",
            "result_summary": str(e)
        })

        save_log_manager({
            "user_id": user_id,
            "account_id": account_id,
            "agent_called": "Insights",
            "action": "Chart Error",
            "result_summary": str(e)
        })
