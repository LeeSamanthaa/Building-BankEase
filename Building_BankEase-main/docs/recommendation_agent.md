# Customer Recommendation System
## Author
**Ayodele Mudavanhu**


## Overview
This repository contains a **hybrid recommendation system** for financial customers that combines **collaborative filtering** and **content-based approaches** to suggest relevant financial products.  
The system analyzes customer transaction patterns and profiles to generate **personalized recommendations**.

## Features
- **Collaborative Filtering** using **Singular Value Decomposition (SVD)**
- **Content-Based Filtering** using **K-Nearest Neighbors (KNN)**
- **Hybrid recommendation generation** combining both approaches
- **Customer profile analysis** and similarity scoring
- **Standardized data preprocessing pipeline**

## Requirements
To run this code, you'll need the following Python packages:
- `pandas`
- `scikit-learn`
- `surprise` (for collaborative filtering)

## Installation
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. Install the required packages:
   ```bash
   pip install pandas scikit-learn surprise
   ```

## Data Preparation
The code expects a CSV file named:
```
cleaned_merged_ieee_bank_10k_version2.csv
```
with the following columns:
- **AccountID** – Unique customer identifier  
- **Merchant** – Merchant information  
- **TransactionAmt** – Transaction amount  
- **CreditScore** – Customer credit score  
- **CustomerAge** – Customer age  
- **CustomerOccupation** – Customer occupation category  
- **PreferredSpendingCategory** – Customer spending preferences  
- **MostInterestedProduct** – Product customer is most interested in  

## System Components

### 1. Collaborative Filtering (SVD)
- Uses the Surprise library's SVD implementation
- Trains on **AccountID–Merchant–TransactionAmt** triples
- Evaluates using 3-fold cross-validation with **RMSE** and **MAE** metrics  
**Configuration:**
  - 50 latent factors  
  - 20 epochs  
  - Learning rate: 0.005  
  - Regularization: 0.02  

### 2. Content-Based Filtering (KNN)
- Creates customer profiles from transaction and demographic data
- One-hot encoding for categorical features
- StandardScaler for feature normalization
- Cosine similarity metric with **5 nearest neighbors**

### 3. Recommendation Generator
- Combines results from both approaches  
- Returns:
  1. Customer's own most interested product
  2. Top recommendation from similar customer #1
  3. Top recommendation from similar customer #2

## Usage
1. Place your transaction data in:
   ```
   cleaned_merged_ieee_bank_10k_version2.csv
   ```
2. Modify the `sample_account_id` variable to test with a specific customer ID  
3. Run the script:
   ```bash
   python recommendation_system.py
   ```

## Output Example
```text
🧪 Generating recommendations for AccountID = 12345

🤝 Collaborative Filtering Recommendations (SVD)
[Cross-validation results...]

🤝 Content-Based Recommendations (KNN)
[KNN model trained...]

🎯 Final Recommendation Output:
1. Customer's own most interested product: Premium Credit Card
2. Similar customer #1's most interested product: Investment Account
3. Similar customer #2's most interested product: Home Loan
```

## Customization
- Adjust SVD hyperparameters in `collaborative_filtering_recommendations`
- Modify customer profile features in `content_based_recommendations`
- Change the number of recommendations in `generate_customer_recommendation`

## Performance Notes
- SVD model includes cross-validation to evaluate performance
- Content-based filtering uses **cosine similarity**, effective for high-dimensional data
- Works well with **sparse transaction data**


### Credits

This recommendation agent was developed by Ayodele Mudavanhu, with integration and documentation support from Sanjiv Shrestha and Madoka Fujii.
