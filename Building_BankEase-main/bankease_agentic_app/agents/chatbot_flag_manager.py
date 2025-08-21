'''
This file defines the Streamlit app for managing chatbot flag requests.
It allows managers to review flagged chatbot responses, make decisions, and add comments.
Once decisions are made, they are updated in the CSV file.
It also provides instructions for managers on how to use the interface.
'''

import streamlit as st
import os
import pandas as pd

# ----- Defining Logging -----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))                # current directory
TRUNK_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))            # Move up one level to reach the trunk
FLAG_FILE = os.path.join(TRUNK_DIR, 'flags', 'flag_chatbot.csv')

def render():

    st.markdown("""
    
    1. **Manager Decision**:  
    - Choose either **Correct** or **Incorrect"** from the dropdown for each row.

    2. **Manager Comments**:  
    - Click inside the comment cell and **type your comment**.
    - After typing, **press `Enter`** the cell to save the comment.

    3. **Saving Your Work**:
    - Once you've filled in all required fields, **click the `Submit All Decisions` button** at the bottom.
    """)

    if os.path.exists(FLAG_FILE):
        df = pd.read_csv(FLAG_FILE)
    else:
        st.warning("No flagged responses found.")
        st.stop()

    df["Manager Decision"] = df["Manager Decision"].astype(str)
    df["Manager Comments"] = df["Manager Comments"].astype(str)
    df["Manager Comments"] = df["Manager Comments"].replace("nan", "")

    # Filter only rows with flag response == False
    df_pending = df[df["Manager Flag"] == False].copy()

    if df_pending.empty:
        st.success("No pending flagged responses.")
        st.stop()

    # filtered_df = df_pending[["Manager Decision", "Manager Comments", "User", 
    #                           "User Question", "Assistant Response", "Timestamp"]]
    filtered_df = df_pending.loc[:, ["Manager Decision", "Manager Comments", "User", 
                                 "User Question", "Assistant Response", "Timestamp"]]

    
    st.set_page_config(layout="wide")
    
    editable_df = st.data_editor(
        filtered_df,
        use_container_width=True,
        num_rows="fixed",  # disables adding/removing rows
        disabled=["User", "User Question", "Assistant Response", "Timestamp"],  # these are read-only
        key="editor",
        column_config={
            "Manager Decision": st.column_config.SelectboxColumn(
                label="Manager Decision",
                options=["", "Correct", "Incorrect"],
                required=False
            )
        }
    )

    if st.button("Submit All Decisions"):
        # Validate and update only rows with a decision
        for idx, row in editable_df.iterrows():
            decision = row["Manager Decision"]
            comment = row["Manager Comments"]
            
            if decision in ["Correct", "Incorrect"]:
                # index of df_pending and df is same as index hasn't been reset
                # original_index = df_pending.index[idx]
                # df.loc[original_index, "Manager Decision"] = decision
                # df.loc[original_index, "Manager Comments"] = comment
                # df.loc[original_index, "Manager Flag"] = True
                df.loc[idx, "Manager Decision"] = decision
                df.loc[idx, "Manager Comments"] = comment
                df.loc[idx, "Manager Flag"] = True
        
        df.to_csv(FLAG_FILE, index=False)
        st.success("Updated Successfully!")
        st.rerun()
