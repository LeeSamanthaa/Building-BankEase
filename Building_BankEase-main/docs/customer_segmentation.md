# Customer Segmentation System
## Author
**Ayodele Mudavanhu**

## Overview
This repository contains a **comprehensive customer segmentation system** that combines **RFM (Recency, Frequency, Monetary) analysis** with **K-Means clustering** to identify distinct customer groups.  
The system helps businesses understand customer behavior patterns and tailor marketing strategies accordingly.

## Features
- **RFM Analysis**: Calculates Recency, Frequency, and Monetary metrics for each customer
- **RFM Scoring**: Quantifies customer value with a composite score (1–15 scale)
- **RFM Segmentation**: Classifies customers into 5 predefined segments (Champions, Loyal, Potential, At Risk, Lost)
- **K-Means Clustering**: Identifies natural customer groupings using machine learning
- **Cluster Interpretation**: Automatically maps clusters to meaningful segment names
- **Model Persistence**: Saves the trained model and scaler for future use

## Requirements
To run this code, you'll need the following Python packages:
- `pandas`
- `numpy`
- `scikit-learn`

## Installation
1. Clone this repository:
   ```bash
   git clone <repo-url>
   cd <repo-folder>
   ```
2. Install the required packages:
   ```bash
   pip install pandas numpy scikit-learn
   ```

## Data Preparation
The code expects a pandas DataFrame with the following columns:
- **AccountID** – Unique customer identifier
- **TransactionDate** – Date of transaction (datetime format)
- **TransactionAmt** – Transaction amount
- **guid** – Transaction identifier (used for frequency calculation)

## Methodology

### 1. RFM Analysis
- **Recency**: Days since last transaction (lower is better)
- **Frequency**: Total number of transactions
- **Monetary**: Total spending amount

### 2. RFM Scoring
- Each metric is divided into 5 quintiles (1–5 score)
- Scores are combined into a composite RFM score (3–15 range)

### 3. Segmentation
- **Champions**: RFM Score ≥ 12
- **Loyal**: RFM Score ≥ 9
- **Potential**: RFM Score ≥ 6
- **At Risk**: RFM Score ≥ 4
- **Lost**: RFM Score < 4

### 4. K-Means Clustering
- Uses 4 clusters (configurable)
- Applies log transformation to handle skewed data
- Standardizes features before clustering
- Automatically interprets clusters based on RFM characteristics

## Usage
1. Prepare your transaction data in a pandas DataFrame
2. Call the `customer_segmentation()` function:
   ```python
   rfm_data, kmeans_model, cluster_summary = customer_segmentation(your_dataframe)
   ```

## Output
The system provides:

- **RFM DataFrame** containing:
  - Raw RFM metrics
  - RFM scores
  - RFM segments
  - Cluster assignments
  - Cluster-based segments

- **Cluster Summary Table** showing:
  - Segment names
  - Average Recency, Frequency, Monetary values
  - Number of customers in each segment

- **Saved Model** (`customer_segmentation_model.pkl`) containing:
  - Trained KMeans model
  - Feature scaler
  - Cluster mapping rules
  - Cluster summary
  - Feature names

### Example Output
```text
📊 Customer Segmentation (KMeans + PCA)

Cluster Summary:
           Segment    Recency  Frequency  Monetary  Count
Cluster                                                  
0         Loyal       45.21      3.12      7.25   2487
1      Champions      12.45      4.87      9.01   1254
2      Potential     102.33      1.95      5.67   1876
3       At Risk     215.78      0.87      3.45    983

💾 Model saved as 'customer_segmentation_model.pkl'

✨ Analysis Complete! ✨
```

## Customization
- Adjust the number of clusters by changing `optimal_k`
- Modify RFM segment thresholds in the `get_rfm_segment` function
- Update cluster naming logic in `map_cluster_to_segment`

## Applications
- Targeted marketing campaigns
- Customer retention programs
- Product recommendation strategies
- Customer lifetime value prediction

### Credits
This Customer Segmentation System was developed by Ayodele Mudavanhu, with integration and documentation support from the Sanjiv Shrestha.
