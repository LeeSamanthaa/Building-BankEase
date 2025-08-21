'''
Answer questions related to only using the transactions of that particular user; access restriction for other user accounts
Checks if given user question is related to transactions of his/her account. If asks questions related to other accounts, doesn't proceed to query generation phase.
generate sql query for given user question -> execute sql query and get response -> use that to generate answer for user
'''

import pandas as pd
import json
from datetime import datetime
import os
from langchain_groq import ChatGroq

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)

# ----- Defining Logging -----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))                      # current directory
TRUNK_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))      # Move up three levels to reach the trunk
DATA_FILE = os.path.join(TRUNK_DIR, 'data', 'gd_chatbot.csv')
DB_PATH = os.path.join(TRUNK_DIR, 'data.db')
LOG_FILE = os.path.join(TRUNK_DIR, 'logs', 'chatbot_logs.jsonl')

def save_log(event: dict):
    os.makedirs("logs", exist_ok=True)
    event["timestamp"] = datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

# ----- db Setup -----

df = pd.read_csv(DATA_FILE)

from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine

engine = create_engine(f"sqlite:///{DB_PATH}")
df.to_sql("data", engine, index=False, if_exists='replace')

db = SQLDatabase(engine=engine)

# ----- Modeling -----

from typing_extensions import TypedDict
from langgraph.graph.message import add_messages
from typing import Literal

class State(TypedDict):
    sql_user_id: str
    sql_account_id: str
    sql_original_question: str
    sql_question: str                                            # user question
    sql_query: str                                               # SQL query generated for the user question
    sql_query_generation_status: Literal["success", "failure"]   # status of query generation
    sql_account_access_status: Literal["write_query", "no_account_access"]
    sql_error_count: int                                         # number of times the query execution has failed
    sql_query_error: str                                         # error message if query generation fails
    sql_result: str                                              # result of executing above generated query
    sql_answer: str                                              # final answer to user question based on result acquired

## ----- Account Access Validation -----

from pydantic import BaseModel, Field
from langgraph.types import Command

class account_access_answer(BaseModel):
    next: Literal["write_query", "no_account_access"] = Field(
        description="Determines whether the user can access the information that he has asked in the question: "
                    "write_query: If the user question is related to his account "
                    "no_account_access: If the question requires accessing records of other account holder."
    )
    reason: str = Field(
        description="Detailed justification for the decision."
    )

def account_access(state: State) -> Command[Literal["write_query", "no_account_access"]]:

    system_message = '''
    You are a data access control assistant in a banking multi-agent system.

    You are given a user question and have access to two variables:
    - user_id: the ID of the current user.
    - account_id: the account ID of the current user.

    Your task is to decide if the user's question can be answered **only using data that belongs to that user_id and account_id**.

    Rules:
    1. If the question asks about **only their own transactions**, account details, balances, summaries, or history — respond with:
        write_query

    2. If the question asks about **data of other users** (explicitly or implicitly), or if answering requires comparison across multiple users/accounts — respond with:
        no_account_access

    3. Do NOT assume that the user is allowed to access data of other users, even if their question is generic or statistical in nature.

    4. Be cautious of questions like:
        - "Who spent the most last month?" → no_account_access
        - "Show me top 10 highest transactions in the bank" → no_account_access
        - "Show my top 10 transactions" → write_query
        - "Compare my balance with other users" → no_account_access
        - "What is my average transaction amount?" → write_query

    Only return **"write_query"** or **"no_account_access"** with no explanation.
    '''

    user_prompt = '''
    User Question: {input}
    Use the following user_id and account_id to determine if allowed or not:
    user_id: {user_id}
    account_id: {account_id}
    '''

    query_prompt_template = ChatPromptTemplate(
        [("system", system_message), ("user", user_prompt)]
    )

    prompt = query_prompt_template.invoke(
        {
            "user_id": state["sql_user_id"],
            "account_id": state["sql_account_id"],
            "input": state["sql_question"],
        }
    )
    response = llm.with_structured_output(account_access_answer).invoke(prompt)

    if response.next=="write_query":
      state["sql_account_access_status"] = "write_query"

    elif response.next=="no_account_access":
      state["sql_account_access_status"] = "no_account_access"

    save_log({
        "user_id": state["sql_user_id"],
        "account_id": state["sql_account_id"],
        "agent_called": "sqlbot: account_access",
        "action": "Determines whether the user can access the information that he has asked",
        "user_query": state["sql_question"],
        "result_summary": f"account_access → {response.next}. Reason: {response.reason}"
    })

    return state

### ----- No Account Access -----

def no_account_access(state: State):
  "Checks if gives question is relevant to that user id and account."

  state["sql_answer"] = "Unauthorized access attempt detected. Only your account information is accessible."

  save_log({
      "user_id": state["sql_user_id"],
      "account_id": state["sql_account_id"],
      "agent_called": "sqlbot: no_account_access",
      "action": "Fallback if SQL query execution fails",
      "user_query": state["sql_question"],
      "result_summary": state["sql_answer"]
  })

  return state

### ----- Account Access Router -----

def account_access_router(state: State):

  if state["sql_account_access_status"] == "write_query":
    save_log({
        "user_id": state["sql_user_id"],
        "account_id": state["sql_account_id"],
        "agent_called": "sqlbot: account_access_router",
        "action": "Routes to write_query, or no_account_access based on sql_account_access_status",
        "user_query": state["sql_question"],
        "result_summary": "Moving to write_query"
    })
    return "write_query"

  elif state["sql_account_access_status"] == "no_account_access":
    save_log({
        "user_id": state["sql_user_id"],
        "account_id": state["sql_account_id"],
        "agent_called": "sqlbot: account_access_router",
        "action": "Routes to write_query, or no_account_access based on sql_account_access_status",
        "user_query": state["sql_question"],
        "result_summary": "Moving to no_account_access"
    })
    return "no_account_access"

## ----- Question to SQL -----

from langchain_core.prompts import ChatPromptTemplate

system_message = """
You are an expert in generating SQL queries for a financial database. 
Given an input question, create a syntactically correct {dialect} query to run to help find the answer.

**Strictly follow this rule:** 
Only return results for the following:
- user_id = {user_id}
- account_id = {account_id}

Never generate queries that access data from other users or accounts.

Use the table schema and the user question below to write the SQL query.

Schema:
{table_info}

If the user question attempts to access multiple users or accounts, ignore that part and return results only for:
user_id = {user_id} AND account_id = {account_id}

Schema highlights:
- TransactionYear: the 4-digit year (e.g., 2024)
- TransactionMonth: an integer from 1 to 12 representing the month. 1 is January, 2 is February, 3 is March and so on.
- TransactionDay: an integer from 1 to 31 representing the day of the month
    Example: To get all transactions that occurred on January 16, 2024, use: WHERE TransactionYear = 2024 AND TransactionMonth = 1 AND TransactionDay = 16
- Category (e.g. ‘Healthcare’, ‘Electronics’, ‘Unknown’)
- Merchant (e.g. ‘MediCare’, ‘ElectroWorld’, ‘Unknown’)
- Channel (one of ‘Online’, ‘ATM’, ‘Branch’)
- CustomerAge (INT, in years)
- CustomerOccupation (one of ‘Retired’, ‘Engineer’, ‘Student’, ‘Teacher’, ‘Doctor’)
- CreditScore (INT, 300–850)
- RiskProfile (one of ‘Low’, ‘Medium’, ‘High’)
- CustomerTenure (INT, years with bank)
- PreferredSpendingCategory (e.g. ‘Electronics’, ‘Entertainment’, …)
- MostInterestedProduct (e.g. ‘Home Loan’, ‘Credit Card’, …)
- IncomeBracket (string ranges: ‘<25K’, ‘25K-50K’,'50K-100K', '150K-200K', '200K-250K', ‘300K-350K’)
  • For comparisons (“>200K”), include all brackets whose lower bound exceeds the threshold (e.g., IN (‘200K-250K’, ‘300K-350K’)).
Respond only with the SQL query.
"""

user_prompt = """Question: {input}
Use the following error information if there is any: {query_error}
"""

query_prompt_template = ChatPromptTemplate(
    [("system", system_message), ("user", user_prompt)]
)

from typing_extensions import Annotated

class QueryOutput(TypedDict):
    """Generated SQL query."""

    generated_sql_query: Annotated[str, ..., "Syntactically valid SQL query."]


def write_query(state: State):
    """Generate SQL query to fetch information."""
    prompt = query_prompt_template.invoke(
        {
            "dialect": db.dialect,
            "table_info": db.get_table_info(),
            "user_id": state["sql_user_id"],
            "account_id": state["sql_account_id"],
            "input": state["sql_question"],
            "query_error": state.get("sql_query_error", ""),
        }
    )
    structured_llm = llm.with_structured_output(QueryOutput)
    result = structured_llm.invoke(prompt)
    state["sql_query"] = result["generated_sql_query"]

    save_log({
        "user_id": state["sql_user_id"],
        "account_id": state["sql_account_id"],
        "agent_called": "sqlbot: write_query",
        "action": "Write SQL query for the user question",
        "user_query": state["sql_question"],
        "result_summary": state["sql_query"]
    })

    return state

## ----- Execute Query ----- 

from langchain_community.tools.sql_database.tool import QuerySQLDatabaseTool

def execute_query(state: State):
    """Execute SQL query and set query_generation_status."""
    execute_query_tool = QuerySQLDatabaseTool(db=db)
    result = execute_query_tool.invoke(state["sql_query"])

    if isinstance(result, str) and result.startswith("Error:"):
        state["sql_query_generation_status"] = "failure"
        state["sql_query_error"] = result
        error_count = state.get("sql_error_count", 0)
        state["sql_error_count"] = error_count + 1
        log_result = f"Query Execution Failed and Error Count is {error_count+1}. Reason for Failure: {result}"
    else:
        state["sql_result"] = result
        state["sql_query_generation_status"] = "success"
        state["sql_query_error"] = ""
        state["sql_error_count"] = 0        # resetting to 0 for future use of attribute
        log_result = f"Query Execution Successful. Result: {result}"

    save_log({
        "user_id": state["sql_user_id"],
        "account_id": state["sql_account_id"],
        "agent_called": "sqlbot: execute_query",
        "action": "Execute the generated SQL query",
        "user_query": state["sql_question"],
        "result_summary": log_result
    })

    return state

## -----  Cannot Answer ----- 

def cannot_answer(state: State):
  "Cannot answer question; return a default answer."
  state["sql_error_count"] = 0        # resetting to 0 for future use of attribute
  state["sql_answer"] = "I'm sorry, but I cannot find the information you're looking for."

  save_log({
      "user_id": state["sql_user_id"],
      "account_id": state["sql_account_id"],
      "agent_called": "sqlbot: cannot_answer",
      "action": "Fallback if SQL query execution fails",
      "user_query": state["sql_question"],
      "result_summary": state["sql_answer"]
  })

  return state

## -----  Routing ----- 

def sql_router(state: State):
  """Routes to generate_answer, cannot_answer or write_query based on query_generation_status."""

  if state["sql_query_generation_status"] == "success":
    save_log({
        "user_id": state["sql_user_id"],
        "account_id": state["sql_account_id"],
        "agent_called": "sqlbot: sql_router",
        "action": "Routes to generate_answer, cannot_answer or write_query based on query_generation_status",
        "user_query": state["sql_question"],
        "result_summary": "Query generated successfully. Moving to generate_answer"
    })
    return "generate_answer"

  elif state["sql_query_generation_status"] == "failure":

    if state["sql_error_count"] < 2:
      save_log({
          "user_id": state["sql_user_id"],
          "account_id": state["sql_account_id"],
          "agent_called": "sqlbot: sql_router",
          "action": "Routes to generate_answer, cannot_answer or write_query based on query_generation_status",
          "user_query": state["sql_question"],
          "result_summary": "Query generation failed and error count is less than maximum allowed. Moving to write_query to try again"
      })
      return "write_query"

    else:
      save_log({
          "user_id": state["sql_user_id"],
          "account_id": state["sql_account_id"],
          "agent_called": "sqlbot: sql_router",
          "action": "Routes to generate_answer, cannot_answer or write_query based on query_generation_status",
          "user_query": state["sql_question"],
          "result_summary": "Query generation failed and error count has reached maximum allowed. Moving to cannot_answer"
      })
      return "cannot_answer"

## -----  Generate Answer -----

def generate_answer(state: State):
    """Answer question using retrieved information as context."""
    prompt = (
        "You are a helpful assistant at a bank who answers a customer's questions about their own transactions.\n"
        f"The customer has User ID: {state['sql_user_id']} and Account ID: {state['sql_account_id']}.\n"
        "You are given their original question and the result of the internal query used to get their transaction data.\n"
        "Using only this result, answer the customer's question clearly and naturally.\n"
        "Use a well-formatted table with clear headers **only if** the customer’s question requires structured data, such as a list of transactions, balances over time, or multiple entries.\n"
        "Otherwise, respond in plain text that reads naturally.\n"
        "Do not mention the SQL query, database, or anything technical.\n"
        "Do not refer to the question or result explicitly (e.g., don't say 'Based on your question...' or 'The result shows...').\n"
        "Avoid starting with generic chatbot phrases like 'Hello there!', 'Hi!', or 'I'd be happy to help you...'.\n"
        "Just give a direct, human-like response as if you're assisting a customer in a natural conversation.\n\n"
        f"Customer Original Question: {state['sql_original_question']}\n"
        f"Customer's Restructured Question: {state['sql_question']}\n"
        f"Result: {state['sql_result']}"
        
    )
    response = llm.invoke(prompt)
    state["sql_answer"] = response.content

    save_log({
        "user_id": state["sql_user_id"],
        "account_id": state["sql_account_id"],
        "agent_called": "sqlbot: generate_answer",
        "action": "Final answer for the user question ",
        "user_query": state["sql_question"],
        "result_summary": state["sql_answer"]
    })

    return state

# ----- Orchestrating with LangGraph -----

from langgraph.graph import StateGraph, START, END

sql_graph_builder = StateGraph(State)

sql_graph_builder.set_entry_point("account_access")
sql_graph_builder.add_node("account_access", account_access)
sql_graph_builder.add_node("no_account_access", no_account_access)
sql_graph_builder.add_conditional_edges(
    "account_access",
    account_access_router,
    {
        "write_query": "write_query",
        "no_account_access": "no_account_access",
    }
)

sql_graph_builder.add_node("write_query", write_query)
sql_graph_builder.add_node("execute_query", execute_query)
sql_graph_builder.add_node("generate_answer", generate_answer)
sql_graph_builder.add_node("cannot_answer", cannot_answer)

sql_graph_builder.add_edge("write_query", "execute_query")

sql_graph_builder.add_conditional_edges(
    "execute_query",
    sql_router,
    {
        "generate_answer": "generate_answer",
        "write_query": "write_query",
        "cannot_answer": "cannot_answer"
    }
)

sql_graph_builder.add_edge("generate_answer", END)
sql_graph_builder.add_edge("cannot_answer", END)
sql_graph_builder.add_edge("no_account_access", END)

sql_graph_customer = sql_graph_builder.compile()
