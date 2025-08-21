# Bankease Agentic App

## Installation

1. **Download this repository**
2. **Ensure Python 3.10 is installed**  
   - Python 3.10 is recommended because later versions (> 3.10) may cause installation issues.
3. Navigate to the project directory:
   ```bash
   cd bankease_agentic_app
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   **Notes:**
   - For the `faiss-cpu` library, a **specific version** is used in `requirements.txt`.  
     Without a version, MacOS may throw the error:  
     ```
     ERROR: Failed building wheel for faiss-cpu
     ```
   - Other libraries do not have specific version constraints.
   
   - On **MacOS**, the provided requirements are usually enough.
   - On **Linux** and **Windows**, you may also need to install:

     ```
     numpy==1.26.4
     faiss-cpu==1.7.4
     ```

     to avoid compatibility errors during installation.
   * Other libraries do not have specific version constraints.
---

## Environment Variables

Create a `.env` file in the project root with the following content:

```env
GROQ_API_KEY="YOUR_GROQ_KEY_HERE"
HF_TOKEN="YOUR_HF_TOKEN_HERE"
```

The `.env` file securely stores API keys and other sensitive configuration values so they are not hardcoded in the source code.
These variables are loaded at runtime by the app.

---

## Running the App

```bash
streamlit run app.py
```

---

## Login Instructions

Two personas are available for login:

1. **Customer** – Client of the bank
2. **Manager** – Bank manager

**Steps:**

1. On the login page, select **Customer** or **Manager** for "Select Role".
2. Refer to the credentials in:
   ```
   bankease_agentic_app/login/login_credentials_encrypted.csv
   ```
   - **Customers**:  
     - 50 customers with usernames: `user_customer1` … `user_customer50`
   - **Manager**:  
     - 1 manager with username: `user_manager`
   - **Password** (all users):  
     ```
     Bankease@123
     ```

---

## Features by Persona

### **Customer**
Access to **5 dashboards**:

1. **AI Assistant**  
   - Ask questions about transactions, bank processes, policies, and FAQs.  
   - Option to *Flag Last Response* to request manager review (**Human-in-the-loop**).
   
2. **Insights**  
   - View transaction insights filtered by Year and Month.
   
3. **Recommendations**  
   - Product recommendations based on:
     - Browsing history
     - Similar customer profiles
   
4. **Fraud Detection**  
   - View fraud vs. non-fraud transaction distribution  
   - Inspect flagged fraud transactions  
   - See manager's decision on specific transactions (**Human-in-the-loop** override)
   
5. **Manager Feedback (AI Assistant)**  
   - View manager’s responses to flagged AI Assistant outputs.

---

### **Manager**
Access to **5 dashboards**:

1. **AI Assistant**  
   - Query transactions for all customers and ask banking-related questions.

2. **Logs**  
   - Filter logs by User ID, Agent, and Date Range.  
   - **Overview View**:
     - High-level view of services used and responses (Insights, Recommendations, Fraud, AI Assistant).  
   - **Detailed View**:
     - Includes sub-processes for deeper insights.
   
3. **KPI Dashboard**  
   - Filter by Year and Month  
   - View performance metrics for all customers.
   
4. **AI Assistant Approval Requests**  
   - Review customer validation requests for AI Assistant outputs.

5. **Fraud Approval Requests**  
   - Review flagged transactions, fraud probabilities, and alerts.  
   - Submit decision (**Flag as Fraud** / **Not Fraud**) and add comments.

---
### **Documentation**

Detailed agent and dashboard documentation can be found in the docs folder:

1. [**AI Assistant (Multi-Agent)**](docs/AI_Assistant_Multiagent.md)  
2. [**Insights Agent**](docs/insights_agent.md)  
3. [**Recommendation Agent**](docs/recommendation_agent.md)  
4. [**Customer Segmentation**](docs/customer_segmentation.md)  
5. [**Fraud Agent**](docs/fraud_agent.md)  
6. [**Fraud Agent Integration**](docs/fraud_agent_integration.md)  
7. [**KPI Dashboard**](docs/kpi_dashboard.md)  

---

### Example Command Flow
```bash
# Navigate to app directory
cd bankease_agentic_app

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

---

## Information regarding other directories

1. **team_AI** 
   - Contains work done by collaborators wrt to insights, recommendations and fraud models.
    A subset of this work has already been incorporated into bankease_agentic_app directory. Not used for running the Bankease agentic app.
    
---

## Acknowledgements

- **Golden Dataset** – Created by Sanjiv Shrestha, Madoka Fujii, Samantha Lee, Anika Rana, Mazhar  
- **FAQ and Bank Policies** – Sunil Thapa  
- **Documentation Files** – Authored by contributors (names mentioned inside each document in `docs/`)
