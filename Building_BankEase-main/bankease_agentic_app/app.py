import streamlit as st
import uuid

from login.login_user import login_user     # importing the function
from logs import preview_logs               # importing the .py file
from agents import chatbot_agent, insights_agents, recommendations_agent, fraud_agent, kpi_dashboard, chatbot_flag_manager, manager_feedback, manager_fraud_feedback, manager_fraud_feedback_view


st.set_page_config(layout="centered")
# Session state init
if 'role' not in st.session_state:
    st.session_state.role = None
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
if 'account_access' not in st.session_state:
    st.session_state.account_access = None
if 'user_access' not in st.session_state:
    st.session_state.user_access = None

# Login
if not st.session_state.logged_in:
    
    role, account_access, user_access = login_user()
    if role and account_access and user_access:
        st.session_state.role = role
        st.session_state.account_access = account_access
        st.session_state.user_access = user_access
        st.session_state.logged_in = True
        st.rerun()          # st.rerun() preserves st.session_state across reruns
else:
    
    st.sidebar.title("Navigation")
    
    if st.session_state.role == 'manager':
        option = st.sidebar.radio("Manager Functions", ["AI Assistant", "Logs","KPI Dashboard", 
                                                        "AI Assistant Approval Requests", "Fraud Approval Requests"])

        if option == "AI Assistant":
           # st.set_page_config(layout="wide")
            st.title("AI Assistant")
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            if "thread_id" not in st.session_state:
                st.session_state.thread_id = str(uuid.uuid4())
            chatbot_agent.render(st.session_state.user_access, st.session_state.account_access, 
                                 st.session_state.chat_history, st.session_state.thread_id, st.session_state.role)
        
        elif option == "Logs":
            st.title("Logs")
            preview_logs.render()
            
        elif option == "KPI Dashboard":
            st.title("KPI Dashboard")
            kpi_dashboard.render()
        
        elif option == "AI Assistant Approval Requests":
            st.title("AI Assistant Approval Requests")
            chatbot_flag_manager.render()
        elif option == "Fraud Approval Requests":
            st.title("Fraud Approval Requests")
            manager_fraud_feedback.render()


    # customer approval requests

    elif st.session_state.role == 'customer':
        option = st.sidebar.radio("Customer Functions", ["AI Assistant", "Insights", "Recommendations", 
                                                         "Fraud Detection", "Manager Feedback (AI Assistant)"])

        if option == "AI Assistant":
            st.title("AI Assistant")
            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            if "thread_id" not in st.session_state:
                st.session_state.thread_id = str(uuid.uuid4())
            chatbot_agent.render(st.session_state.user_access, st.session_state.account_access, 
                                 st.session_state.chat_history, st.session_state.thread_id, st.session_state.role)
        
        elif option == "Insights":  
            st.title("Insights on transactions")    
            insights_agents.render(st.session_state.user_access, st.session_state.account_access)
        
        elif option == "Recommendations":
            st.title("Product Recommendations")
            recommendations_agent.render(st.session_state.user_access, st.session_state.account_access)
        
        elif option == "Fraud Detection":
            st.title("Fraud Detection")
            # st.subheader("📋 Manager Feedback (Fraud Detection)")
            manager_fraud_feedback_view.render(st.session_state.user_access)

            st.markdown("---")

            # Then show the fraud analytics
            fraud_agent.render(st.session_state.user_access, st.session_state.account_access)


        elif option == "Manager Feedback (AI Assistant)":
            st.title("Manager Feedback (AI Assistant)")
            manager_feedback.render(st.session_state.user_access)
            st.set_page_config(layout="centered")
        


    st.sidebar.button("Logout", on_click=lambda: st.session_state.update({"logged_in": False, "role": None,
                                                                          "user_access": None, "account_access": None,
                                                                          "chat_history": [], "thread_id": None}))
