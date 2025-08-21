# -*- coding: utf-8 -*-
# Uninstall the default numpy version in Colab
!pip uninstall numpy -y

# Install the required numpy version
!pip install numpy==1.26.4

# Verify the numpy version
import numpy as np
print(f"Numpy version after installation: {np.__version__}")

# Core Libraries
import pandas as pd
import pickle # Changed from joblib to pickle for potentially wider compatibility
import json
import warnings
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (roc_auc_score, precision_recall_curve)
from sklearn.feature_selection import VarianceThreshold
import lightgbm as lgb
from tqdm import tqdm
# Suppress warnings for a cleaner output
warnings.filterwarnings("ignore")

print(f"Numpy version after installation: {np.__version__}")

# --- PART 1: CONFIGURATION AND DATA LOADING ---

print("\n## PART 1: Configuration and Data Loading")

# --- File and Column Configuration ---
ORIGINAL_DATA_PATH = 'cleaned_merged_ieee_bank_10k_version2.csv'
TARGET_COLUMN = 'isFraud'
DATE_COLUMN_CANDIDATES = ['TransactionDate', 'Date', 'TransactionDT', 'month_year', 'timestamp']
UID_COLUMNS_ORIGINAL = ['guid', 'UserID', 'AccountID']
TRANSACTION_AMOUNT_COL = 'TransactionAmt'

# --- Model and Feature Engineering Configuration ---
N_SPLITS = 5
TOP_N_FEATURES = 100 # This parameter is not directly used in the current version of the code, but kept for context

# --- Chatbot-Specific Configurations ---
CHATBOT_CONFIG = {
    'high_risk_threshold': 0.75,
    'medium_risk_threshold': 0.35,
    'low_risk_threshold': 0.15,
    'confidence_levels': {
        'very_high': 0.90, 'high': 0.75, 'medium': 0.50, 'low': 0.25
    },
    'alert_messages': {
        'very_high': " HIGH FRAUD RISK! We've flagged this transaction for your immediate review. Please confirm if this was you.",
        'high': " Potential Fraud Alert: This transaction shows unusual patterns. Please review your recent activity.",
        'medium': " For Your Review: This transaction has been flagged. If it looks unfamiliar, please contact support.",
        'low': " Transaction appears normal. Logged for monitoring."
    }
}

print(f"Loading dataset from: {ORIGINAL_DATA_PATH}")
try:
    df = pd.read_csv(ORIGINAL_DATA_PATH, dtype={'UserID': str, 'AccountID': str, 'guid': str})
    print(f"Dataset loaded successfully. Shape: {df.shape}")

    actual_date_column = None
    for col in DATE_COLUMN_CANDIDATES:
        if col in df.columns:
            try:
                # Use a larger sample to be more robust
                test_series = pd.to_datetime(df[col].sample(n=min(len(df), 1000), random_state=42), errors='coerce')
                if not test_series.isna().all():
                    actual_date_column = col
                    break
            except Exception:
                continue

    if actual_date_column is None:
        print("WARNING: No suitable date column found. Generating a dummy 'TransactionDate'.")
        df['TransactionDate'] = pd.date_range('2024-01-01', periods=len(df), freq='H')
        actual_date_column = 'TransactionDate'

    DATE_COLUMN = actual_date_column
    print(f"Using date column: {DATE_COLUMN}")

except FileNotFoundError:
    print(f"ERROR: File '{ORIGINAL_DATA_PATH}' not found. Please ensure the CSV is in the correct directory.")
    # Exit is not ideal in a multi-part program or server environment, better to raise an exception
    raise FileNotFoundError(f"Required data file not found: {ORIGINAL_DATA_PATH}")

# --- PART 2: LEAK-FREE DATA PREPROCESSING AND FEATURE ENGINEERING ---

print("\n## PART 2: Leak-Free Data Preprocessing and Feature Engineering")

df_processed = df.copy()

df_processed[DATE_COLUMN] = pd.to_datetime(df_processed[DATE_COLUMN], errors='coerce')
df_processed = df_processed.dropna(subset=[DATE_COLUMN]).sort_values(by=DATE_COLUMN).reset_index(drop=True)
print(f"Sorted data chronologically. Shape: {df_processed.shape}")

imputation_values = {}
# Ensure we don't try to impute the target column
numerical_cols = [col for col in df_processed.select_dtypes(include=np.number).columns if col != TARGET_COLUMN]
categorical_cols_obj = [col for col in df_processed.select_dtypes(include=['object']).columns if col not in UID_COLUMNS_ORIGINAL]

for col in numerical_cols:
    if df_processed[col].isnull().any():
        median_val = df_processed[col].median()
        df_processed[col].fillna(median_val, inplace=True)
        imputation_values[col] = {'type': 'numerical', 'value': float(median_val)}

for col in categorical_cols_obj:
    if df_processed[col].isnull().any():
        mode_val = df_processed[col].mode()[0] if not df_processed[col].mode().empty else 'UNKNOWN'
        df_processed[col].fillna(mode_val, inplace=True)
        imputation_values[col] = {'type': 'categorical', 'value': mode_val}
print("- Missing value imputation completed.")

def create_leak_free_features(df_input, user_col, amount_col, date_col):
    """
    Creates leak-free expanding and rolling features for a given user column.
    Features are shifted by 1 to ensure they only use past information.
    Reverted to a more stable rolling feature creation approach to avoid errors.
    """
    df_temp = df_input.copy()
    df_temp.sort_values([user_col, date_col], inplace=True)
    df_temp.reset_index(drop=True, inplace=True)

    # Expanding Features
    df_temp[f'{user_col}_count_expanding'] = df_temp.groupby(user_col).cumcount()
    df_temp[f'{user_col}_amount_mean_expanding'] = df_temp.groupby(user_col)[amount_col].expanding().mean().reset_index(level=0, drop=True)

    for col in [f'{user_col}_count_expanding', f'{user_col}_amount_mean_expanding']:
        df_temp[col] = df_temp.groupby(user_col)[col].shift(1)
    df_temp[f'{user_col}_count_expanding'].fillna(0, inplace=True)
    df_temp[f'{user_col}_amount_mean_expanding'].fillna(df_temp[amount_col].mean(), inplace=True)

    # --- FIX APPLIED HERE: Reverted to the original, stable rolling logic ---
    rolling_windows = ['30min', '1H', '2H', '6H', '12H', '24H', '3D', '7D', '14D', '30D']
    rolling_aggs = ['mean', 'sum', 'std', 'min', 'max', 'median'] # Kept the reduced list for stability

    for window in tqdm(rolling_windows, desc="Creating Rolling Features"):
        # The corrected method: use .rolling() with the 'on' parameter, which works reliably within a groupby object
        rolling_agg_features = df_temp.groupby(user_col).rolling(window, on=date_col, min_periods=1)[amount_col].agg(rolling_aggs)

        for agg in rolling_aggs:
            # Need to align the aggregated results with the original dataframe
            # Use .values to extract the data and align it
            df_temp[f'{user_col}_amount_{agg}_{window}'] = rolling_agg_features[agg].values

        # Correctly compute and align the rolling count
        rolling_count = df_temp.groupby(user_col).rolling(window, on=date_col, min_periods=1)[amount_col].count()
        df_temp[f'{user_col}_txn_count_{window}'] = rolling_count.values

    # Now, shift ALL new rolling features by 1 in one go.
    # This ensures no data leakage from the current transaction
    new_rolling_cols = [col for col in df_temp.columns if any(f'_{window}' in col for window in rolling_windows)]
    for col in new_rolling_cols:
        df_temp[col] = df_temp.groupby(user_col)[col].shift(1)
        df_temp[col].fillna(0, inplace=True)
    # --- END OF FIX ---

    # Time Since Last Event Features
    # Renamed the column to be more explicit and avoid potential conflicts
    if 'ProductCD' in df_temp.columns:
        df_temp['time_since_last_user_product_txn'] = df_temp.groupby([user_col, 'ProductCD'])[date_col].diff().dt.total_seconds().shift(1).fillna(0)
    if 'Merchant' in df_temp.columns:
        df_temp['time_since_last_user_merchant_txn'] = df_temp.groupby([user_col, 'Merchant'])[date_col].diff().dt.total_seconds().shift(1).fillna(0)

    # Ratio Features
    df_temp['txn_amt_to_avg_ratio'] = df_temp[amount_col] / (df_temp[f'{user_col}_amount_mean_expanding'] + 1e-6)
    df_temp['txn_amt_to_avg_ratio'].fillna(0, inplace=True)
    for window in ['24H', '7D']:
        max_col = f'{user_col}_amount_max_{window}'
        min_col = f'{user_col}_amount_min_{window}'
        if max_col in df_temp.columns:
            df_temp[f'txn_amt_to_max_ratio_{window}'] = df_temp[amount_col] / (df_temp[max_col] + 1e-6)
            df_temp[f'txn_amt_to_max_ratio_{window}'].fillna(0, inplace=True)
        if min_col in df_temp.columns:
            df_temp[f'txn_amt_to_min_ratio_{window}'] = df_temp[amount_col] / (df_temp[min_col] + 1e-6)
            df_temp[f'txn_amt_to_min_ratio_{window}'].fillna(0, inplace=True)

    df_temp.reset_index(drop=True, inplace=True)
    return df_temp

actual_uid_cols = [col for col in UID_COLUMNS_ORIGINAL if col in df_processed.columns]
if TRANSACTION_AMOUNT_COL in df_processed.columns and actual_uid_cols:
    df_processed = create_leak_free_features(df_processed, user_col=actual_uid_cols[0],
                                             amount_col=TRANSACTION_AMOUNT_COL, date_col=DATE_COLUMN)
    if len(actual_uid_cols) > 1:
        print(f"Note: Using '{actual_uid_cols[0]}' as the primary UID for advanced rolling features. Consider extending for other UIDs if needed.")
    print("- Leak-free aggregation features completed.")
else:
    print("WARNING: Transaction amount column or UID columns not found. Skipping leak-free aggregation features.")

df_processed['hour'] = df_processed[DATE_COLUMN].dt.hour
df_processed['day_of_week'] = df_processed[DATE_COLUMN].dt.dayofweek
df_processed['day_of_month'] = df_processed[DATE_COLUMN].dt.day
print("- Time-based features created.")

if 'ProductCD' in df_processed.columns and 'Merchant' in df_processed.columns and actual_uid_cols:
    df_processed['user_product_interaction'] = df_processed[actual_uid_cols[0]].astype(str) + '_' + df_processed['ProductCD'].astype(str)
    df_processed['user_merchant_interaction'] = df_processed[actual_uid_cols[0]].astype(str) + '_' + df_processed['Merchant'].astype(str)
    print("- Interaction features created.")

label_encoders = {}
all_categorical_cols = [col for col in categorical_cols_obj if col in df_processed.columns] + [col for col in actual_uid_cols if col in df_processed.columns]

if 'user_product_interaction' in df_processed.columns:
    all_categorical_cols.append('user_product_interaction')
if 'user_merchant_interaction' in df_processed.columns:
    all_categorical_cols.append('user_merchant_interaction')

for col in all_categorical_cols:
    df_processed[col] = df_processed[col].fillna('UNKNOWN').astype(str)
    le = LabelEncoder()
    # FIX: Correctly converting the numpy array to a pandas Series before concatenation
    le.fit(pd.concat([pd.Series(df_processed[col].unique()), pd.Series(['UNKNOWN'])]).astype(str))
    df_processed[col] = le.transform(df_processed[col].astype(str))
    label_encoders[col] = le
print("- Categorical features encoded.")

df_processed.fillna(0, inplace=True)
df_processed.replace([np.inf, -np.inf], 0, inplace=True)
print("- Final NaN and Inf values handled.")

# --- PART 3: FEATURE SELECTION AND PREPARATION ---

print("\n## PART 3: Feature Selection and Preparation")
exclude_cols_final = [DATE_COLUMN, TARGET_COLUMN] + UID_COLUMNS_ORIGINAL
feature_cols = [col for col in df_processed.columns if col not in exclude_cols_final]
X = df_processed[feature_cols].copy()
y = df_processed[TARGET_COLUMN].copy()

# Filter out features with no variance before training
# This is a good practice to reduce noise and potential issues with estimators
constant_features = [col for col in X.columns if X[col].nunique() <= 1]
if constant_features:
    print(f"Removing constant features: {constant_features}")
    X.drop(columns=constant_features, inplace=True)

selector = VarianceThreshold(threshold=0.001)
X_selected = X[X.columns[selector.fit(X).get_support(indices=True)]]
print(f"- Remaining features after variance threshold: {X_selected.shape[1]}")

final_feature_names = list(X_selected.columns)

# --- PART 4: MODEL TRAINING WITH TIME SERIES CROSS-VALIDATION ---

print("\n## PART 4: Model Training with Time Series Cross-Validation")
train_feature_names = final_feature_names
X_train_final = X_selected[train_feature_names]

tscv = TimeSeriesSplit(n_splits=N_SPLITS)
scale_pos_weight = (y == 0).sum() / (y == 1).sum() if (y == 1).sum() > 0 else 1

lgbm_params = {
    'objective': 'binary',
    'metric': 'auc',
    'n_estimators': 3000,
    'learning_rate': 0.005,
    'num_leaves': 60,
    'seed': 42,
    # Explicitly set n_jobs=1 to avoid multiprocessing issues in certain environments
    'n_jobs': 1,
    'verbose': -1,
    'colsample_bytree': 0.6,
    'subsample': 0.6,
    'reg_alpha': 0.1,
    'reg_lambda': 0.1,
    'scale_pos_weight': scale_pos_weight,
    'boosting_type': 'gbdt' # Explicitly set boosting type
}

models = {'lgb': []}
oof_predictions = {'lgb': np.zeros(len(X_train_final))}

print("Training LightGBM models...")
for fold, (train_idx, val_idx) in enumerate(tqdm(tscv.split(X_train_final, y), total=N_SPLITS, desc="Training Folds")):
    X_train_fold, X_val_fold = X_train_final.iloc[train_idx], X_train_final.iloc[val_idx]
    y_train_fold, y_val_fold = y.iloc[train_idx], y.iloc[val_idx]

    lgb_model = lgb.LGBMClassifier(**lgbm_params)
    lgb_model.fit(X_train_fold, y_train_fold,
                  eval_set=[(X_val_fold, y_val_fold)],
                  eval_metric='auc',
                  callbacks=[lgb.early_stopping(200, verbose=False)])

    val_pred_lgb = lgb_model.predict_proba(X_val_fold)[:, 1]
    oof_predictions['lgb'][val_idx] = val_pred_lgb
    auc_lgb = roc_auc_score(y_val_fold, val_pred_lgb)
    print(f"--- Fold {fold + 1}/{N_SPLITS} AUC: {auc_lgb:.6f} ---")
    models['lgb'].append(lgb_model)

overall_auc = roc_auc_score(y, oof_predictions['lgb'])
print(f"\nOverall Model Performance (OOF AUC): {overall_auc:.6f}")
print("""** How to Interpret the AUC Score **
An AUC of 0.5 is random guessing, and 1.0 is a perfect model.
An AUC between 0.8 and 0.9 (like in the individual folds) is considered 'Excellent'.
The lower Overall OOF AUC suggests the model performs well on stable data but may struggle with concept drift over longer time periods.
This is common in fraud detection as fraud patterns evolve.""")

# --- PART 5: CHATBOT-SPECIFIC THRESHOLD OPTIMIZATION ---

print("\n## PART 5: Chatbot-Specific Threshold Optimization")

def optimize_chatbot_thresholds(y_true, y_pred_proba, desired_recall=0.80):
    """
    Optimizes the threshold to achieve a desired recall,
    prioritizing precision at that recall level.
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    idx = np.where(recall >= desired_recall)[0]

    if len(idx) > 0:
        # Find the index with the highest precision among those meeting the recall target
        best_idx = idx[np.argmax(precision[idx])]
    else:
        # If no threshold meets the recall target, use the one with the highest recall
        best_idx = np.argmax(recall)

    # Ensure the index is within bounds of the thresholds array
    best_threshold = thresholds[min(best_idx, len(thresholds) - 1)]

    # Handle the case where the best index is out of range for precision/recall arrays
    precision_at_best_threshold = precision[best_idx]
    recall_at_best_threshold = recall[best_idx]

    return {'threshold': best_threshold, 'precision': precision_at_best_threshold, 'recall': recall_at_best_threshold}

# Filter predictions where a model was actually trained for that time period
oof_mask = oof_predictions['lgb'] != 0
if oof_mask.sum() > 0:
    optimal_metrics = optimize_chatbot_thresholds(y[oof_mask], oof_predictions['lgb'][oof_mask], desired_recall=0.80)
    CHATBOT_CONFIG['low_risk_threshold'] = optimal_metrics['threshold']

    print(f"Optimal Threshold for desired recall of ~80%:")
    print(f"  - Threshold: {optimal_metrics['threshold']:.4f}")
    print(f"  - Precision: {optimal_metrics['precision']:.4f}")
    print(f"  - Recall: {optimal_metrics['recall']:.4f}")
    print(f"\nUpdated chatbot low-risk (alert) threshold to {optimal_metrics['threshold']:.4f}")
    print("""** How to Interpret Precision and Recall **
- A Recall of {recall:.2f} means: We will successfully catch {recall:.0%} of all actual fraudulent transactions.
- A Precision of {precision:.2f} means: When we do send an alert, there is a {precision:.1%} chance it is for a genuinely fraudulent transaction.
This is a classic trade-off: to catch more fraud (high recall), we must accept more false positives (lower precision).""".format(recall=optimal_metrics['recall'], precision=optimal_metrics['precision']))
else:
    print("Warning: No OOF predictions available to optimize thresholds. Using default configuration.")

# --- PART 6: CHATBOT SIMULATION FUNCTION ---

print("\n## PART 6: Chatbot Simulation Function")

def generate_chatbot_response(probability, config):
    """Generates a human-readable chatbot message based on fraud probability."""
    if probability >= config['confidence_levels']['very_high']:
        return config['alert_messages']['very_high']
    elif probability >= config['confidence_levels']['high']:
        return config['alert_messages']['high']
    # Use the optimized threshold for the 'medium' alert level
    elif probability >= config['low_risk_threshold']:
        return config['alert_messages']['medium']
    else:
        return config['alert_messages']['low']

print("- Chatbot response function is ready.")

# --- PART 7: DEPLOYMENT ARTIFACTS AND FINAL DEMONSTRATION ---

print("\n## PART 7: Deployment Artifacts and Final Demonstration")

deployment_artifacts = {
    'models': models['lgb'],
    'feature_names': train_feature_names,
    'imputation_values': imputation_values,
    'label_encoders': label_encoders,
    'chatbot_config': CHATBOT_CONFIG,
    'uid_cols': actual_uid_cols
}

# Switched from joblib to pickle requested
try:
    with open('fraud_detection_artifacts.pkl', 'wb') as f:
        pickle.dump(deployment_artifacts, f, protocol=pickle.HIGHEST_PROTOCOL)
    print("- Deployment artifacts saved to 'fraud_detection_artifacts.pkl' using pickle.")
except Exception as e:
    print(f"ERROR: Failed to save deployment artifacts with pickle. {e}")
    # Fallback to joblib if pickle fails for some reason
    try:
        import joblib
        joblib.dump(deployment_artifacts, 'fraud_detection_artifacts_fallback.pkl')
        print("- Saved with joblib as a fallback to 'fraud_detection_artifacts_fallback.pkl'")
    except Exception as e_joblib:
        print(f"ERROR: Failed to save with joblib as well. {e_joblib}")


print("\n--- FINAL DEMONSTRATION FOR A SINGLE USER ---")

user_id_col_for_demo = 'UserID'
if user_id_col_for_demo not in actual_uid_cols:
    if actual_uid_cols:
        user_id_col_for_demo = actual_uid_cols[0]
    else:
        user_id_col_for_demo = None

if user_id_col_for_demo and user_id_col_for_demo in df_processed.columns and user_id_col_for_demo in label_encoders:
    user_fraud_counts = df_processed.groupby(user_id_col_for_demo)[TARGET_COLUMN].value_counts().unstack(fill_value=0)
    suitable_users_encoded = user_fraud_counts[(user_fraud_counts[1] > 0) & (user_fraud_counts[0] > 0)]

    if not suitable_users_encoded.empty:
        chosen_user_encoded = suitable_users_encoded.index[0]
        user_id_encoder = label_encoders[user_id_col_for_demo]
        chosen_user_original = user_id_encoder.inverse_transform([chosen_user_encoded])[0]

        print(f"Simulating for original user: {chosen_user_original} (Encoded as: {chosen_user_encoded})")
        user_transactions_original = df.loc[df[user_id_col_for_demo] == chosen_user_original].sort_values(by=DATE_COLUMN)

        results = []
        for index, transaction_row in user_transactions_original.iterrows():
            transaction_dict = transaction_row.to_dict()

            # Apply imputation values
            for col, details in imputation_values.items():
                if col in transaction_dict and pd.isna(transaction_dict[col]):
                    transaction_dict[col] = details['value']

            # Apply label encoding
            for col, le in label_encoders.items():
                if col in transaction_dict:
                    val = transaction_dict[col]
                    # Handle unseen categories gracefully
                    try:
                        encoded_val = le.transform([str(val)])[0]
                    except ValueError:
                        encoded_val = le.transform(['UNKNOWN'])[0] # Use 'UNKNOWN' as fallback
                    transaction_dict[col] = encoded_val

            # Create a DataFrame for prediction
            # Reindex to ensure all features are present, with NaNs filled by 0
            temp_df = pd.DataFrame([transaction_dict]).reindex(columns=train_feature_names, fill_value=0)

            # Ensure data types are consistent for prediction
            for col in temp_df.columns:
                if temp_df[col].dtype == 'object':
                    # This should not happen with the current preprocessing, but as a safeguard
                    temp_df[col] = pd.to_numeric(temp_df[col], errors='coerce').fillna(0)

            # Predict using the ensemble of models
            try:
                probabilities = [model.predict_proba(temp_df)[:, 1][0] for model in models['lgb']]
                probability = np.mean(probabilities)
            except Exception as e:
                print(f"Prediction failed for a transaction: {e}")
                probability = 0.0

            should_alert = bool(probability >= CHATBOT_CONFIG['low_risk_threshold'])
            chatbot_message = generate_chatbot_response(probability, CHATBOT_CONFIG)

            results.append({
                'Date': pd.to_datetime(transaction_row[DATE_COLUMN]).strftime('%Y-%m-%d'),
                'Amount': f"${transaction_row[TRANSACTION_AMOUNT_COL]:.2f}",
                'ActualFraud': 'Yes' if transaction_row[TARGET_COLUMN] == 1 else 'No',
                'FraudProbability': f"{probability:.1%}",
                'Alert': 'YES' if should_alert else 'NO',
                'ChatbotResponse': chatbot_message
            })

        results_df = pd.DataFrame(results)
        print(results_df.to_string())
    else:
        print("Could not find a user with both fraud and non-fraud transactions for the demo. Please check your data.")
else:
    print(f"The specified user ID column '{user_id_col_for_demo}' was not found or not encoded. Cannot run demo.")

print("\nPIPELINE EXECUTION COMPLETE.")

# Re-run python version command for verification
print("\nVerifying Python version:")
try:
    import sys
    print(sys.version)
except Exception as e:
    print(f"Could not retrieve python version: {e}")
