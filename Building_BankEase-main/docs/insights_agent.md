# Insights Visualization Agent
## Author
**Priyanjali Patel**

The **Insights Visualization Agent** is a Streamlit-based module that generates interactive financial dashboards for users based on their transaction data. It allows filtering by year and month, and provides charts summarizing spending patterns, merchant activity, and customer profile metrics.

---

## 📂 File Location

```
agents/insights.py
```

---

## ⚙️ Function Overview

### `render(user_id, account_id)`

Main function responsible for:

* Reading and filtering transaction data.
* Generating interactive visualizations.
* Logging user interactions.

---

## 🔍 How It Works

### 1. **Data Loading**

* Loads `cleaned_merged_ieee_bank_10k_chatbot_version2.csv` from the `/data` folder.
* Filters rows for the given `account_id`.
* Extracts `Year` and `Month` from the `TransactionDate` column.

---

### 2. **Filters**

The user can filter data by:

* **Year** (dropdown, descending order).
* **Month** (dropdown, ascending order).

---

### 3. **Visualizations**

#### **📊 Monthly Spending Summary**

* Bar chart showing the total debit spend for the selected month.

#### **📈 Daily Spending Pattern**

* Daily debit spending bar chart with a red dashed line showing the **average monthly spend**.

#### **🔍 Transaction Type Distribution**

* Pie chart of Debit vs Credit transactions.

#### **🧾 Spending by Category**

* Pie chart of spending distribution by `Category` (if available).

#### **🏪 Most Frequent Merchants**

* Bar chart showing the top 5 merchants by transaction frequency.

#### **💸 High-Value Merchants**

* Bar chart showing the top 5 merchants by total debit spend.

#### **🛰 Transaction Channels**

* Bar chart showing counts of transactions per `Channel`.

#### **👤 Profile Summary**

* Displays:

  * Customer tenure in years.
  * Credit score.

---

### 4. **Logging**

The agent logs:

* **User actions** into `logs/chatbot_logs.jsonl`.
* **Manager view logs** into `logs/manager_logs.jsonl`.

Logs store:

* `user_id`
* `account_id`
* `agent_called`
* `action`
* `result_summary`
* Timestamp (ISO format)

---

## 🖼 Example Output

![Monthly Spending Chart](/images/insights_monthly_spend.png)
![Daily Pattern](/images/insights_daily_spend.png)
![Category Pie Chart](/images/insights_category_pie.png)

---

## 💡 Notes

* If no transaction data exists for the selected filters, the app shows a warning.
* LLM functionality (`INSIGHTS_PROMPT`) is imported but **currently unused** in this script.
* Uses **Matplotlib** and **Seaborn** for charts.

---
## Credits

This engine was developed by Priyanjali Patel and Madoka Fujii, with integration and documentation support from Olajumoke Ojikutu and Amit Sarkar.
