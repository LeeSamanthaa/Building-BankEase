'''
This module handles user login functionality for the BankEase application.
It verifies user credentials against stored encrypted data and returns the user's role, account, and username upon successful login.
'''

import streamlit as st
import pandas as pd
import bcrypt
import ast
import os
import time

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(CURRENT_DIR, "login_credentials_encrypted.csv")

def login_user():
    st.title("Login")

    role = st.selectbox("Select Role", ["Select", "Customer", "Manager"])
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if role == "Select" or not username or not password:
            st.error("Please fill all fields and select a valid role.")
            return None, None, None

        else:
            cred_df = pd.read_csv(CSV_PATH)

            for i in range(cred_df.shape[0]):
                user_name_i = cred_df.loc[i, "User Name"]
                pass_encrypted_i = cred_df.loc[i, "password_encrypted"]
                stored_hash_i = ast.literal_eval(pass_encrypted_i)
                role_stored = cred_df.loc[i, "UserAccess"]
                account = cred_df.loc[i, "Account"]
                user = cred_df.loc[i, "User"]

                if username==user_name_i and bcrypt.checkpw(password.encode('utf-8'), stored_hash_i) and role==role_stored:
                    st.success("Login Successful!")
                    with st.spinner("Processing..."):
                        time.sleep(1)   
                    if role=="Manager":
                        return "manager", account, user
                    elif role=="Customer":
                        return "customer", account, user
                            
            st.error("Invalid Credentials!")    
            return None, None, None

    return None, None, None   # when button is not clicked; function return a single None by default
