# 📾 Golden Dataset - Matching and Merging IEEE Fraud Dataset with Bank Transactions

## Overview

This dataset merges two key sources:

* **Fraud Engine**: IEEE-CIS Fraud Detection Dataset
* **Recommendation/Insights Engine**: Bank Transactions Dataset

Records from both datasets were matched **one-to-one** based on user behavior patterns and linked via a shared synthetic `guid`. This enables unified user-level context for fraud, recommendation, and insight engines.

---

## Objectives

* Align user records across different engines (Fraud, Recommendation, Insights)
* Provide a consistent schema with shared identifiers
* Prepare clean, realistic data for demos, dashboards, and modeling

---

## Matching Logic

* **Features Used**: `Amount`, `Card1`, `Transaction Time`
* **Normalization**: StandardScaler applied to align feature scale
* **Matching Method**: Greedy nearest-neighbor based on Euclidean distance
* **Constraint**: One-to-one matching (no duplicates)
* **GUID**: Generated using `uuid.uuid4()` per matched pair

---

## File Details

* `cleaned_merged_guid_df.csv`
  Merged dataset with suffixes applied: `_ieee` for fraud source, `_bank` for recommendation source

* **Example Columns**:

  * `TransactionID_ieee`, `TransactionTS_ieee`
  * `TransactionID_bank`, `TransactionTS_bank`
  * `guid` (common identifier)

* **Non-null Filter**:

  * Columns with **<90% completeness** were removed to ensure data quality

---

## Schema Overview

| Column Name             | Description                   | Source      |
| ----------------------- | ----------------------------- | ----------- |
| guid                    | Shared synthetic GUID         | Generated   |
| TransactionAmt          | Fraud transaction amount      | IEEE        |
| card1                   | Encoded card ID (user)        | IEEE        |
| TransactionTS\_ieee     | Timestamp (UNIX-style)        | IEEE        |
| TransactionAmount       | Transaction amount            | Bank        |
| card1\_sim              | Simulated card1 via AccountID | Bank        |
| TransactionTS\_bank     | Timestamp (UNIX-style)        | Bank        |


---
## Data Card

| Column Name               | Data Type | Definition                                                                                         |
| ------------------------- | --------- | -------------------------------------------------------------------------------------------------- |
| `TransactionID_ieee`      | int64     | Unique transaction ID in the IEEE Fraud Detection dataset. Numeric unique key.                     |
| `isFraud`                 | int64     | Target variable indicating whether the transaction is fraudulent. 0 = normal, 1 = fraud.           |
| `TransactionDT`           | int64     | Numeric timestamp representing elapsed seconds since a reference point defined in the dataset.     |
| `TransactionAmt`          | float64   | Transaction amount (usually in USD). Float type, representing the spending amount.                 |
| `ProductCD`               | object    | Code indicating the product or service category (e.g., 'W'=Wallet, 'C'=Credit, 'S'=Service, etc.). |
| `TransactionTS_ieee`      | float64   | Actual transaction timestamp in datetime format derived from the IEEE dataset.                     |
| `guid`                    | object    | Synthetic user identifier used to join records across multiple datasets (IEEE + bank).             |
| `TransactionID_bank`      | object    | Unique transaction ID in the bank transaction dataset.                                             |
| `AccountID`               | object    | Unique identifier for the bank account associated with each transaction.                           |
| `TransactionAmount`       | float64   | Amount transferred or spent in the transaction.                                                    |
| `TransactionDate`         | object    | Transaction date from the bank side (calendar-based).                                              |
| `TransactionType`         | object    | Type of transaction (e.g., deposit, withdrawal, payment, transfer, etc.).                          |
| `Location`                | object    | Geographic location where the transaction took place (e.g., city, state).                          |
| `DeviceID`                | object    | Unique identifier of the device used by the user to make the transaction.                          |
| `IP Address`              | object    | IP address from which the user made the transaction.                                               |
| `MerchantID`              | object    | Unique identifier of the merchant involved in the transaction.                                     |
| `Channel`                 | object    | Channel through which the transaction was made (e.g., mobile app, web, ATM, in-branch).            |
| `CustomerAge`             | float64   | Age of the customer.                                                                               |
| `CustomerOccuoation`      | object    | Customer's occupation.                                 |
| `TransactionDuration`     | float64   | Duration of the transaction (in seconds or minutes).                                               |
| `LoginAttempts`           | int64     | Number of login attempts made on the same day.                                                     |
| `AccountBalance`          | float64   | Account balance at the time of the transaction.                                                    |
| `PreciousTransactionDate` | object    | Date of the most recent transaction prior to the current one.                                      |
| `TransactionTS_bank`      | float64   | Normalized timestamp of the transaction in the bank dataset.                                       |



For questions or contributions, please feel free to let me know.
