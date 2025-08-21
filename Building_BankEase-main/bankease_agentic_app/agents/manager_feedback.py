'''
<<<<<<< HEAD

=======
This module is designed to manage and display feedback from managers on flagged responses by agents. 
It reads from a CSV file that contains flagged responses and displays them in a structured format. If no flagged responses are found, it shows a warning message to the user.
This module is essential for reviewing and managing feedback from managers, ensuring that flagged responses are addressed appropriately.
>>>>>>> multiagent-sql-auth
'''

import streamlit as st
import os
import pandas as pd

# ----- Defining Logging -----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))                # current directory
TRUNK_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))            # Move up one level to reach the trunk
FLAG_FILE = os.path.join(TRUNK_DIR, 'flags', 'flag_chatbot.csv')

def render(user_id):

    if os.path.exists(FLAG_FILE):
        df = pd.read_csv(FLAG_FILE)
        df = df[df["User"]==user_id]
        if df.shape[0] == 0:
            st.warning("No flagged responses yet.")
            st.stop()  
        df = df.rename(columns={"Manager Decision": "Manager Response"})
        st.dataframe(
            df[["User Question", "Assistant Response", "Manager Response", "Manager Comments", "Timestamp"]],
            use_container_width=True
        )
    else:
        st.warning("No flagged responses yet.")
        st.stop()
