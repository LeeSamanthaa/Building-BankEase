'''
This module provides functionality to preview logs for the Manager persona.
It allows users to view logs in either an overview or detailed format, with options to filter by user ID, agent, and date range.
'''

import streamlit as st
import json
import pandas as pd

def render():

    if "logs_mode" not in st.session_state:
        st.session_state.logs_mode = None
        
    st.write('\n')

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Overview"):
            st.session_state.logs_mode = "Overview"
    with col2:
        if st.button("Detailed"):
            st.session_state.logs_mode = "Detailed"

    # Ensure a mode is selected
    if st.session_state.logs_mode is None:
        st.warning("Please select a mode to continue.")
        st.stop()

    # detailed table view
    if st.session_state.logs_mode == "Detailed":

        try:
            with open("logs/chatbot_logs.jsonl", "r", encoding="utf-8") as f:
                logs = [json.loads(line) for line in f]
            df_logs = pd.DataFrame(logs)
            df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])

            # --- Filters ---
            col1, col2, col3 = st.columns(3)

            with col1:
                user_filter = st.selectbox("Filter by User ID", ["All"] + sorted(df_logs["user_id"].dropna().unique().tolist()))

            with col2:
                agent_filter = st.selectbox("Filter by Agent", ["All"] + sorted(df_logs["agent_called"].dropna().unique().tolist()))

            with col3:
                date_range = st.date_input("Filter by Date Range", [])

            # --- Apply filters ---
            if user_filter != "All":
                df_logs = df_logs[df_logs["user_id"] == user_filter]

            if agent_filter != "All":
                df_logs = df_logs[df_logs["agent_called"] == agent_filter]

            if len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)  # Include date
                df_logs = df_logs[(df_logs["timestamp"] >= start_date) & (df_logs["timestamp"] < end_date)]

            df_logs = df_logs.sort_values("timestamp", ascending=False)

            # ----- Show table -----

            df_display = df_logs[["timestamp", "user_id", 
                                  "agent_called", "result_summary"]].rename(columns={"timestamp": "Time Stamp",
                                                                                "user_id": "User ID",
                                                                                "agent_called": "Agent",
                                                                                "result_summary": "Response"
                                                                                    }
                                                                            )

            with st.container():
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    height=500,
                    hide_index=True
                )

        except FileNotFoundError:
            st.warning("No logs available yet.")
    
    # overview table view
    elif st.session_state.logs_mode == "Overview":
    
        try:

            with open("logs/manager_logs.jsonl", "r", encoding="utf-8") as f:
                logs = [json.loads(line) for line in f]
            df_logs = pd.DataFrame(logs)
            df_logs["timestamp"] = pd.to_datetime(df_logs["timestamp"])

            # --- Filters ---
            col1, col2, col3 = st.columns(3)

            with col1:
                user_filter = st.selectbox("Filter by User ID", ["All"] + sorted(df_logs["user_id"].dropna().unique().tolist()))

            with col2:
                agent_filter = st.selectbox("Filter by Agent", ["All"] + sorted(df_logs["agent_called"].dropna().unique().tolist()))

            with col3:
                date_range = st.date_input("Filter by Date Range", [])

            # --- Apply filters ---
            if user_filter != "All":
                df_logs = df_logs[df_logs["user_id"] == user_filter]

            if agent_filter != "All":
                df_logs = df_logs[df_logs["agent_called"] == agent_filter]

            if len(date_range) == 2:
                start_date = pd.to_datetime(date_range[0])
                end_date = pd.to_datetime(date_range[1]) + pd.Timedelta(days=1)  # Include date
                df_logs = df_logs[(df_logs["timestamp"] >= start_date) & (df_logs["timestamp"] < end_date)]

            df_logs = df_logs.sort_values("timestamp", ascending=False)

            # ----- Show table -----

            df_display = df_logs[["timestamp", "user_id", "agent_called", 
                                  "action", "result_summary"]].rename(columns={"timestamp": "Time Stamp",
                                                                                "user_id": "User ID",
                                                                                "agent_called": "Agent",
                                                                                "action": "Action",
                                                                                "result_summary": "Response"
                                                                                }
                                                                        )

            with st.container():
                st.dataframe(
                    df_display,
                    use_container_width=True,
                    height=500,
                    hide_index=True
                )

        except FileNotFoundError:
            st.warning("No logs available yet.")
