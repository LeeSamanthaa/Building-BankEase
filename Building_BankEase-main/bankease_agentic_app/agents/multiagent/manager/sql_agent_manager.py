'''
Answer questions related to any user/account in the db; no access restriction for the user
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
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))                # current directory
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
    sql_error_count: int                                         # number of times the query execution has failed
    sql_query_error: str                                         # error message if query generation fails
    sql_result: str                                              # result of executing above generated query
    sql_answer: str                                              # final answer to user question based on result acquired

## ----- Question to SQL -----

from langchain_core.prompts import ChatPromptTemplate

system_message = """
Given an input question, create a syntactically correct {dialect} query to run to help find the answer.

Never query for all the columns from a specific table, only ask for a the few relevant columns given the question.

Pay attention to use only the column names that you can see in the schema description. Be careful to not query for columns that do not exist. Also, pay attention to which column is in which table.

Only use the following tables:
{table_info}

Schema highlights:
- TransactionYear: the 4-digit year (e.g., 2024)
- TransactionMonth: an integer from 1 to 12 representing the month
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
        "You are a banking assistant responding to a manager's queries about customer transactions.\n"
        "The manager has access to all user IDs and account IDs.\n"
        "Given the manager's question and the result of the internal SQL query used to retrieve the relevant data,\n"
        "answer the question clearly and professionally.\n"
        "Use a well-formatted table with clear headers **only if** the customer’s question requires structured data, such as a list of transactions, balances over time, or multiple entries.\n"
        "Otherwise, respond in plain text that reads naturally.\n"
        "Do not mention SQL queries, databases, or how the data was retrieved.\n"
        "Avoid phrases like 'Hello there!', 'I'm happy to help...', or anything overly formal or robotic.\n"
        "Give a direct, informative, human-like answer as if responding to a manager's internal query.\n\n"
        f"Manager's Original Question: {state['sql_original_question']}\n"
        f"Manager's Restructured Question: {state['sql_question']}\n"
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

from langgraph.graph import StateGraph, END

sql_graph_builder = StateGraph(State)

sql_graph_builder.set_entry_point("write_query")
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

sql_graph_manager = sql_graph_builder.compile()
