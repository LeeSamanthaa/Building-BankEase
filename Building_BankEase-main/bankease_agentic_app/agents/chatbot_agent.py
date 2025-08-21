'''
Chat Interface for Manager and Customer Login.
User asks for a question which is answered using app_manager and app_customer which
are multi-agent graphs defined within the multiagent.manager.multi_agent_manager & 
multiagent.customer.multi_agent_customer respectively.
'''

import streamlit as st
import os
from datetime import datetime
import json
import pandas as pd

from agents.multiagent.manager.multi_agent_manager import app_manager
from agents.multiagent.customer.multi_agent_customer import app_customer 

# ----- Defining Logging -----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))                # current directory
TRUNK_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))            # Move up one level to reach the trunk
LOG_FILE = os.path.join(TRUNK_DIR, 'logs', 'chatbot_logs.jsonl')
MANAGER_LOG_FILE = os.path.join(TRUNK_DIR, 'logs', 'manager_logs.jsonl')
FLAG_FILE = os.path.join(TRUNK_DIR, 'flags', 'flag_chatbot.csv')

def save_log_manager(event: dict):
    os.makedirs("logs", exist_ok=True)
    event["timestamp"] = datetime.now().isoformat()
    with open(MANAGER_LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

def save_log(event: dict):
    os.makedirs("logs", exist_ok=True)
    event["timestamp"] = datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

def render(user_id, account_id, chat_history_passed, thread_id_passed, access_role):

    st.write('\n')

# --- Initialize session state ---
    st.session_state.last_user_id = user_id
    st.session_state.last_account_id = account_id
    st.session_state.chat_history = chat_history_passed
    st.session_state.thread_id = thread_id_passed

    if "last_response" not in st.session_state:
        st.session_state.last_response = None

    if "last_user_input" not in st.session_state:
        st.session_state.last_user_input = None
    if access_role=="customer":
        if st.button("🚩 Flag Last Response"):
            try:
                if st.session_state.last_response and st.session_state.last_user_input:
                    flagged_data = {
                        "User": st.session_state.last_user_id,
                        "User Question": st.session_state.last_user_input,
                        "Assistant Response": st.session_state.last_response,
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Manager Flag": False,
                        "Manager Decision": "",
                        "Manager Comments": ""
                    }

                    flagged_df = pd.DataFrame([flagged_data])
                    if os.path.exists(FLAG_FILE):
                        flagged_df.to_csv(FLAG_FILE, mode="a", header=False, index=False)
                    else:
                        flagged_df.to_csv(FLAG_FILE, mode="w", header=True, index=False)

                    st.success("Response flagged and saved!")
                else:
                    st.warning("No response to flag yet.")
                
            except Exception as e:
                st.warning("Apologies, something went wrong. Please try again later.")

    # writes the past query-answer pair in the chatbot screen; stored in st.session_state.chat_history
    for role, msg in st.session_state.chat_history:
        st.chat_message(role).write(msg)

    if user_input := st.chat_input("What is your question?"):
        
        try:
            st.chat_message("user").write(user_input)
            st.session_state.chat_history.append(("user", user_input))
            
            if access_role=="manager":
                # app_manager calls the multi-agent graph defined for the manager
                answer = app_manager.invoke({"question": user_input, "user_id": st.session_state.last_user_id, 
                                    "account_id": st.session_state.last_account_id},
                                    config={"configurable": {"thread_id": st.session_state.thread_id}})
                response = answer["messages"][-1].content
                with st.chat_message("assistant"):
                    st.write(response)

            elif access_role=="customer":
                # app_customer calls the multi-agent graph defined for the customer
                answer = app_customer.invoke({"question": user_input, "user_id": st.session_state.last_user_id, 
                                    "account_id": st.session_state.last_account_id},
                                    config={"configurable": {"thread_id": st.session_state.thread_id}})

                response = answer["messages"][-1].content

                with st.chat_message("assistant"):
                    st.write(response)

                st.session_state.last_user_input = user_input
                st.session_state.last_response = response       
                   
            save_log_manager({
                "user_id": user_id,
                "account_id": account_id,
                "agent_called": "Chatbot",
                "action": f"Responding to User Query: {user_input}",
                "result_summary": response
            })

            st.session_state.chat_history.append(("assistant", response))

        except Exception as e:
            error_message = "Apologies, something went wrong. Please try again later."
            with st.chat_message("assistant"):
                st.write(error_message)

            save_log({
                "user_id": user_id,
                "account_id": account_id,
                "agent_called": "Chatbot",
                "action": "Error Executing the user Query",
                "user_query": user_input,
                "result_summary": str(e)
            })

            save_log_manager({
                "user_id": user_id,
                "account_id": account_id,
                "agent_called": "Chatbot",
                "action": f"Error Executing the user Query: {user_input}",
                "result_summary": str(e)
            })
