# Documentation: Fraud Engine Integration
## Author
**Samantha Lee & Luis Mancio**

## Purpose

This module is part of the fraud detection system.
Its goal is to load the pre-trained Machine Learning model, process a user’s transaction history, and display visual insights to help identify potential fraudulent activities.

The general idea:

* Load the trained model (`fraud_detection_artifacts_np.pkl`), which contains:

  * The prediction model.
  * The list of variables it uses.
  * Preprocessing rules (filling missing values, encoding text, etc.).
* Get the user’s transactions from the dataset (`cleaned_merged_ieee_bank_10k_version2.csv`).
* Apply the same preprocessing used during model training.
* Predict fraud probability for each transaction.
* Visualize results so analysts or managers can understand risks.

---

## Code Structure

### 1. Library Imports

The script uses:

* **streamlit** → to create an interactive web dashboard.
* **pandas**, **numpy** → for data manipulation.
* **matplotlib**, **seaborn** → for statistical visualizations.
* **pickle** → to load the saved model and configurations.
* **json** & **datetime** → to save usage logs.

---

### 2. Path and File Definitions

It defines absolute paths to:

* The trained model (`fraud_detection_artifacts_np.pkl`).
* The transaction dataset.
* Log files for recording actions and results.

---

### 3. Support Functions

**`save_log(event)`**

* Stores a JSON entry in a log file with user activity (who ran the tool, what they looked for, what was found).

**`save_log_manager(event)`**

* Does the same, but logs are saved separately for management review.

---

### 4. Feature Engineering — `create_leak_free_features`

This function generates extra features from the raw data but avoids data leakage (the model never sees “future” information when making a prediction).

Steps include:

* **Cumulative metrics:** number of previous transactions per user, running average of transaction amounts.
* **Rolling time-window metrics:** average, sum, and count of transactions in the last:

  * 30 minutes
  * 1 hour
  * 24 hours
  * 7 days
* **Time since last transaction** (by product or merchant).
* **Leak prevention:** `.shift(1)` ensures the model only sees past data when predicting.

---

### 5. Fraud Prediction — `predict_fraud_ensemble`

This function:

* Calculates fraud probability from each model since the model used cross validation.
* Averages all probabilities of each model.
* Assigns **1 (fraud)** or **0 (non-fraud)** based on a risk threshold defined in the model.

---

### 6. Main Function — `render(user_id, account_id)`

This is the main workflow.

#### Step 1 — Load model and artifacts

* Opens `fraud_detection_artifacts_np.pkl`.
* Extracts:

  * Models (`models`).
  * Feature names (`feature_names`).
  * Missing value rules (`imputation_values`).
  * Encoders for categorical columns (`label_encoders`).
  * Chatbot configuration (`chatbot_config`).
  * User identifier columns (`uid_cols`).

#### Step 2 — Load data and filter by account

* Opens the dataset CSV.
* Checks that `AccountID` and `isFraud` columns exist.
* Keeps only transactions for the given `account_id`.

#### Step 3 — Preprocess data

* Generates features with `create_leak_free_features()`.
* Fills missing values using `imputation_values`.
* Encodes categorical values into numbers using `label_encoders`.
* Aligns the DataFrame with the model’s expected columns (`feature_names`).

#### Step 4 — Make predictions

* Gets probabilities and fraud predictions with `predict_fraud_ensemble`.
* Adds **FraudProbability** and **Prediction** columns to the processed data.

#### Step 5 — Visualize results

* **Summary metrics:** total transactions vs detected fraud cases.
* **Bar chart:** fraud vs non-fraud counts.
* **Channel analysis:** fraud distribution by transaction channel (if available).
* **Detailed fraud list:** optional table showing all detected fraudulent transactions.

#### Step 6 — Save logs

* Writes activity summary to both the user log and the manager log.

---

### Credits
This fraud agent was developed by Samantha Lee, with integration and documentation support from Luis Mancio and Priyanjali Patel with additional contributions from Ayodele Mudavanhu, Sanjiv Shrestha, Jayash, Amit Sarkar, Ramy Othman and Olajumoke Ojikutu.
