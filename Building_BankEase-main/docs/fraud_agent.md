# **Fraud Agent Documentation**
## Author
**Samantha Lee**

## **Purpose**

The primary goal of this notebook is to train a robust fraud detection model and generate deployment artifacts (`fraud_detection_artifacts.pkl`).
These artifacts contain all components needed to make real-time fraud predictions in a separate application.

---

## **Key Sections and Functionality**

### **Part 1: Configuration and Data Loading**

* **Objective:** Set up global parameters and load the raw transaction dataset.
* **Details:**

  * Defines file paths (`ORIGINAL_DATA_PATH`), target column (`isFraud`), date columns, and user ID columns.
  * Initializes `CHATBOT_CONFIG` with default alert thresholds.
  * Loads the dataset `cleaned_merged_ieee_bank_10k_version2.csv`.
  * Automatically identifies the correct date column and converts it to datetime objects.
* **Libraries Used:** `pandas`

---

### **Part 2: Leak-Free Data Preprocessing and Feature Engineering**

* **Objective:** Clean, transform, and enrich data to create machine learning-ready features without data leakage.
* **Key Steps:**

  1. **Chronological Sorting:** Sort `df_processed` by the identified date column to preserve time order.
  2. **Missing Value Imputation:**

     * Numerical → median fill
     * Categorical → mode fill
  3. **`create_leak_free_features` Function:**

     * **Expanding Features:** Cumulative transaction counts, running averages per user.
     * **Rolling Features:** Mean, sum, std, min, max, median, skew, kurtosis over sliding time windows (`30min`, `1H`, `7D`) per user.
     * **Time Since Last Event:** Seconds since last transaction for each user by product and merchant.
     * **Ratio Features:** Ratios like `txn_amt_to_avg_ratio`.
     * **Leak-Free Mechanism:** All time-dependent features shifted by 1 to ensure no future data leaks.
  4. **Additional Features:** Hour, day of week, day of month.
  5. **Interaction Features:** User-product and user-merchant combinations.
  6. **Label Encoding:** All categorical variables converted to numeric.
  7. **Final NaN/Inf Handling:** Replace remaining NaN/Inf with 0.
* **Libraries Used:** `pandas`, `numpy`, `sklearn.preprocessing.LabelEncoder`, `tqdm`

---

### **Part 3: Feature Selection and Preparation**

* **Objective:** Remove unhelpful features to improve performance.
* **Method:**

  * `VarianceThreshold(threshold=0.001)` drops features with near-constant values.
* **Libraries Used:** `sklearn.feature_selection.VarianceThreshold`

---

### **Part 4: Model Training with Time Series Cross-Validation**

* **Objective:** Train a fraud detection model with realistic temporal validation.
* **Model:** `lightgbm.LGBMClassifier`
* **Method:**

  * **TimeSeriesSplit (n\_splits=5):** Training data always precedes validation data chronologically.
  * **Ensemble Training:** One LightGBM model per fold; predictions averaged.
  * **Class Imbalance Handling:** `scale_pos_weight` adjusted for fraud rarity.
  * **Evaluation Metric:** AUC (Area Under ROC Curve) with early stopping.
  * **Concept Drift Note:** A drop in AUC over time may indicate shifting fraud patterns.
* **Libraries Used:** `lightgbm`, `sklearn.model_selection.TimeSeriesSplit`, `sklearn.metrics.roc_auc_score`

---

### **Part 5: Chatbot-Specific Threshold Optimization**

* **Objective:** Tune fraud probability thresholds for chatbot alerts.
* **Method:**

  * `optimize_chatbot_thresholds` uses `precision_recall_curve` to find the threshold for **80% recall**.
  * Balances high recall with acceptable precision.
  * Updates `CHATBOT_CONFIG` with the optimized `low_risk_threshold`.
* **Libraries Used:** `sklearn.metrics.precision_recall_curve`, `numpy`

---

### **Part 6: Chatbot Simulation Function**

* **Objective:** Convert fraud probabilities into readable chatbot messages.
* **Method:**

  * `generate_chatbot_response` applies thresholds (`very_high`, `high`, `low_risk_threshold`) to categorize risk.
* **Libraries Used:** Pure Python logic

---

### **Part 7: Deployment Artifacts and Final Demonstration**

* **Objective:** Package and validate the trained system for real-time use.
* **Steps:**

  1. Save models, features, encoders, imputation rules, and chatbot config to `fraud_detection_artifacts.pkl` via `joblib.dump()`.
  2. Simulate predictions for a selected user:

     * Apply preprocessing, feature engineering, prediction, and chatbot messaging.
     * Confirm full pipeline correctness.
* **Libraries Used:** `joblib`, `pandas`, `numpy`

---

## **Code Documentation: Real-Time Inference / Simulation (finalized\_fraud\_model.py)**

### **1. Load Deployment Artifacts**

* Load `.pkl` file with:

  * Models
  * Feature names
  * Imputation values
  * Label encoders
  * Chatbot config
  * User ID columns
* Includes error handling for missing files.
* **Libraries Used:** `joblib`

### **2. `create_leak_free_features` Function**

* Replicates training feature engineering exactly.
* Ensures consistency between training and real-time inference.
* Prevents data leakage with `.shift(1)`.
* **Libraries Used:** `pandas`, `numpy`

### **3. `predict_fraud_ensemble` Function**

* Runs all ensemble models.
* Returns mean fraud probability.
* **Libraries Used:** `numpy`

### **4. `generate_chatbot_response` Function**

* Converts fraud probability to chatbot alert message.
* Uses predefined thresholds from artifacts.

### **5. Main Simulation Block**

* **Flow:**

  1. Define `new_transaction_data`.
  2. Load historical transactions for the same user.
  3. Combine historical and new data.
  4. Apply preprocessing and feature engineering.
  5. Align features with model expectations.
  6. Predict fraud probability.
  7. Generate chatbot message.
* **Output:** Prints transaction details, fraud probability, and chatbot response.

### Credits
This fraud agent was developed by Samantha Lee, with integration and documentation support from Luis Mancio and Priyanjali Patel with additional contributions from Ayodele Mudavanhu, Sanjiv Shrestha, Jayash, Amit Sarkar, Ramy Othman and Olajumoke Ojikutu.
