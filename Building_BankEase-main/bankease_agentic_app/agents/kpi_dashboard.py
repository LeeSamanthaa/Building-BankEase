import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import math
import squarify
import os


def render():
    @st.cache_data
    def load_data():
        # Go up one folder from 'agents' → reach project root
        project_root = os.path.dirname(os.path.dirname(__file__))
        
        # Then go into 'data' folder
        file_path = os.path.join(
            project_root,
            "data",
            "cleaned_merged_ieee_bank_10k_chatbot_version2.csv"
        )

        # Ensure it's absolute
        file_path = os.path.abspath(file_path)

        # Load the CSV
        df = pd.read_csv(file_path, encoding="utf-8")
        df['TransactionDate'] = pd.to_datetime(df['TransactionDate'], errors='coerce')
        return df
    
    df = load_data()
    df["Year"] = df["TransactionDate"].dt.year
    df["Month"] = df["TransactionDate"].dt.month



    # --- Filter by year and month ---
    st.subheader("Filter by year and month")

    col1, col2 = st.columns(2)

    with col1:
        selected_year = st.selectbox("Select year", sorted(df["Year"].unique(), reverse=True))

    with col2:
        selected_month = st.selectbox("Select month", sorted(df["Month"].unique()))


    # --- Filered ---
    df_filtered = df[(df["Year"] == selected_year) & (df["Month"] == selected_month)]

    st.markdown(f"Showing results from **{selected_month:02d}/{selected_year}**")
    st.write(f"{len(df_filtered)} found transactions.")

    # --- Graphics ---



    # --- Spend by Age  group ---
    st.subheader("Spend by Age Group")
        
    #Spend by Age group
    # 1. unique_users
    unique_users = df_filtered[['UserID', 'CustomerAge', 'AvgMonthlySpend']].drop_duplicates(subset='UserID')

    # 2. Group Ages
    unique_users['AgeRange'] = pd.cut(unique_users['CustomerAge'], bins=10)

    # 3. Monthly spend by age
    spend_by_range = unique_users.groupby('AgeRange')['AvgMonthlySpend'].mean().sort_index()

    # 4. labels y axis

    labels = [
        f"{math.ceil(r.left)}–{math.floor(r.right)}" for r in spend_by_range.index
    ]

    # 5. Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x=spend_by_range.values, y=labels, ax=ax, palette="viridis")

    ax.set_xlabel("Average Monthly Spend ($)")
    ax.set_ylabel("Customer Age (years)")
    plt.tight_layout()
    plt.show()

    st.pyplot(fig)  # Muestra la figura en Streamlit
   


    #----Channel Performance-----

    st.subheader("Channel Performance")

    #1. Count channel
    channel_counts = df_filtered['Channel'].value_counts()

    def format_autopct_channelp(pct):
        total=channel_counts.sum()
        value = pct *total /100
        return f"{pct:.1f}%\n({value:.0f} transactions)"


    # 2. Pie chart
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.pie(
        channel_counts.values,
        labels=channel_counts.index,
        autopct=format_autopct_channelp,
        startangle=90,
        colors=['#66c2a5', '#fc8d62'],
        wedgeprops=dict(edgecolor='white')
    )

    plt.tight_layout()
    st.pyplot(fig)
    


    #----Spend by Category------
    st.subheader("Spend by Category")
    # 1. Group by Categories
    spend_by_category = df_filtered.groupby('Category')['TransactionAmt'].sum().sort_values(ascending=False)


    # 2. Create Treemap
    fig, ax = plt.subplots(figsize=(8, 8))  # figura más cuadrada
    squarify.plot(
        sizes=spend_by_category.values,
        label=[f"{cat}\n${val:,.0f}" for cat, val in zip(spend_by_category.index, spend_by_category.values)],
        color=sns.color_palette("magma", len(spend_by_category)),
        text_kwargs={'fontsize': 10, 'weight': 'bold','color': 'white'},
        ax=ax,
        norm_x=1, norm_y=1  # fuerza área cuadrada
    )

    plt.axis('off')
    plt.tight_layout()
    st.pyplot(fig)

    #---- Merchants Activity------
    st.subheader("Merchants triggering most activity")
    
    # 1. Group by merchant and sum transactionAmt
    spend_by_merchant = df_filtered.groupby('Merchant')['TransactionAmt'].sum().sort_values(ascending=False)

    # 2. Top 10 merchants
    top_merchants = spend_by_merchant.head(10)

    # 3. Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_merchants.values, y=top_merchants.index, ax=ax, palette="Blues_d")

    # 4. Labels
    ax.set_xlabel("Total Spend ($)")
    ax.set_ylabel("Merchant")
    plt.tight_layout()
    st.pyplot(fig)


    #-----Risk Profile----
    st.subheader("High Risk Profile")
    
     # 1. Defining order
    risk_order = ['Low', 'Medium', 'High']

    # 2. Order cathegory
    df_filtered['RiskProfile'] = pd.Categorical(df_filtered['RiskProfile'], categories=risk_order, ordered=True)

    # 3. Count users
    risk_users = df_filtered.groupby('RiskProfile')['UserID'].nunique()
    

    # 4. Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=risk_users.values, y=risk_users.index, ax=ax, palette="viridis")

    # 5. Labels
    ax.set_xlabel("Number of users")
    ax.set_ylabel("Risk Profile")
    plt.tight_layout()
    

    # 3. Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=risk_users.values, y=risk_users.index, ax=ax, palette="viridis")

    # 4. Labels
    ax.set_xlabel("Number of users")
    ax.set_ylabel("Risk Profile")
    plt.tight_layout()
    st.pyplot(fig)




    #---- Average Monthly Spend by Customer Occupation------
    st.subheader("Average Monthly Spend by Customer Occupation")
    
    # 1. Group by merchant and sum transactionAmt
    spend_by_occupation = df_filtered.groupby('CustomerOccupation')['TransactionAmt'].sum().sort_values(ascending=False)

    # 2. Top 10 merchants
    top_occupation = spend_by_occupation.head(10)

    # 3. Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=top_occupation.values, y=top_occupation.index, ax=ax, palette="Set2")

    # 4. Labels
    ax.set_xlabel("Average Monthly Spend ($)")
    ax.set_ylabel("Customer Occupation")
    plt.tight_layout()
    st.pyplot(fig)

# 📊 Customer Insights & Segmentation
    def customer_segmentation(df):
        print("\n📊 Customer Segmentation")
        
        # RFM Analysis
        current_date = df['TransactionDate'].max() + pd.Timedelta(days=1)
        
        rfm = df.groupby('AccountID').agg({
            'TransactionDate': lambda x: (current_date - x.max()).days,  # Recency
            'guid': 'count',  # Frequency
            'TransactionAmt': 'sum'      # Monetary
        }).reset_index()
        
        rfm.columns = ['AccountID', 'Recency', 'Frequency', 'Monetary']
        
        # RFM Scoring (lower recency is better, higher frequency/monetary is better)
        rfm['R_Score'] = pd.qcut(rfm['Recency'], 5, labels=[5, 4, 3, 2, 1])
        rfm['F_Score'] = pd.qcut(rfm['Frequency'], 5, labels=[1, 2, 3, 4, 5])
        rfm['M_Score'] = pd.qcut(rfm['Monetary'], 5, labels=[1, 2, 3, 4, 5])
        
        rfm['RFM_Score'] = rfm['R_Score'].astype(int) + rfm['F_Score'].astype(int) + rfm['M_Score'].astype(int)
    
        # RFM Segments
        def get_rfm_segment(row):
            if row['RFM_Score'] >= 12:
                return 'Champions'
            elif row['RFM_Score'] >= 9:
                return 'Loyal'
            elif row['RFM_Score'] >= 6:
                return 'Potential'
            elif row['RFM_Score'] >= 4:
                return 'At Risk'
            else:
                return 'Lost'
    
        rfm['Segment'] = rfm.apply(get_rfm_segment, axis=1)

        return rfm

    # Run customer segmentation
    rfm = customer_segmentation(df)

    # Plot RFM segments
    st.subheader("Customer Segments by RFM Score")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=rfm['Segment'].value_counts().values, y=rfm['Segment'].value_counts().index, ax=ax, palette="viridis")    
    plt.title('Customer Segments by RFM Score')
    plt.ylabel('Segment')
    plt.xlabel('Number of Customers')
    plt.xticks(rotation=45)
    plt.show()
    plt.tight_layout()
    st.pyplot(fig)


    
    






