'''
Defines the app_manager used in chatbot_manager.
User question -> restructure_query agent that restructures the question considering past conversations.
supervisor agent -> directs given query to sqlbot (answer transaction-related questions), ragbot (bank policies, faqs-related questions),
or if given question is irrelevant to the scope of the chatbot
sqlbot: Answer questions related to any user/account in the db; no access restriction for the user
ragbot: use RAG to retrieve most relevant info for the question
'''

import os
import json
from datetime import datetime
from langchain_groq import ChatGroq
from agents.multiagent.manager.sql_agent_manager import sql_graph_manager
from agents.multiagent.manager.rag_agent_manager import rag_chain

from dotenv import load_dotenv
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)

# ----- Defining Logging -----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))                # current directory
TRUNK_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))      # Move up three levels to reach the trunk
LOG_FILE = os.path.join(TRUNK_DIR, 'logs', 'chatbot_logs.jsonl')

def save_log(event: dict):
    os.makedirs("logs", exist_ok=True)
    event["timestamp"] = datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

from typing import TypedDict, Annotated, Sequence, List, Literal
from pydantic import BaseModel, Field
from langgraph.types import Command
from langgraph.graph import add_messages, StateGraph, START, END, MessagesState
from langchain_core.messages import HumanMessage, AIMessage

class AgentState(TypedDict):
    account_id: str
    user_id: str
    original_question: str
    question: str
    messages: Annotated[list, add_messages]
    conversation_summary: str
    memory_index: int
    evaluate_answer_status: Literal["Yes", "No"]

# ----- Defining the Nodes -----

## ----- Summarize Conversation (not a node) -----

def summarize_conversation(past_conversation, memory_index, conv_summary):
    
    latest_conv = past_conversation[memory_index:]
    if not latest_conv:     # if latest_conv is empty (no recent conv to summarize), return summary and memory index as is
      return conv_summary, memory_index

    formatted = []
    for msg in latest_conv:
        role = msg.name or msg.__class__.__name__.replace("Message", "").lower()
        formatted.append(f"{role}: {msg.content}")
    formatted_conv = "\n".join(formatted)

    """
    <New conversation>
    user_question: How many transactions are there for UserID U001?
    supervisor: The user question requires accessing transactional data, specifically the total count of transactions associated with a specific UserID, which falls under SQLBot's capabilities.
    """

    system_prompt = (
        '''You are a concise assistant that incrementally summarizes conversations.
        Ensure the summary is always less than 300 words.
        During summarization ensure that the latest conversations are given more importance than the ealier conversations.
        Don't start the summary with phrases such as "Here is the updated summary" '''
    )

    user_prompt = (f'''
        Given the existing summary: {conv_summary},
        Update the summary by incorporating this new conversations: {formatted_conv}
    ''')

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    updated_summary = llm.invoke(messages).content
    memory_index = memory_index + len(latest_conv)    # update memory_index with new conversations summarized

    return updated_summary, memory_index

## ----- Restructure Question -----

def restructure_question(state: AgentState):

    system_prompt = ('''
    You are an assistant in a multi-agent system with two agents:
    1. SQLBot: Handles questions requiring access to transactional or customer-specific bank records.
    2. RAGBot: Handles questions about bank policies, services, FAQs, and general information.

    Your task is to slightly rephrase the user's question only to improve clarity, without changing its original intent or meaning.

    Strictly follow these rules:
    - If the user query is relevant to banking transactions or banking policies/services, you may reword it for clarity.
    - You must **preserve all numerical values, constraints, filters, timeframes, and keywords** exactly as they appear in the original question (e.g., "last 5", "after January", "only withdrawals"). 
    - If the user query is irrelevant, offensive, nonsensical, or outside the banking domain, return it **unchanged**.
    - Do NOT add assumptions, timeframes (e.g., "last 6 months"), filters (e.g., "credit and debit transactions"), or extra details not explicitly mentioned by the user.
    - Do NOT summarize, expand, or alter the scope.
    - Do NOT make the question sound polite or fix tone/attitude unless it's a banking-relevant query.
    - Do NOT prefix your response with any explanation or justification.
    - Use the Summary of Past Conversations only for context to restructure question. Don't change the very meaning of user question with information from past conversations.
                     
    Example:
    User Question: My account id?
    **Correct** Restructured Question: What is the user's Account ID? 
    **Wrong** Restructured Question: What is the user's User ID? -> This is wrong since the user is asking about Account ID and is being restructured asking for User ID.
    
    Just return the restructured (or original) question as a single sentence.
    ''')

    question = state["question"]
    state["original_question"] = question
    past_conversation = state["messages"]
    conv_summary = state.get("conversation_summary", "")
    memory_index = state.get("memory_index", 0)

    summary_conversations, updated_memory_index = summarize_conversation(past_conversation, memory_index, conv_summary)

    user_prompt = f'''
        User Question: {question}
        Summary of Past Conversations: {summary_conversations}
    '''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = llm.invoke(messages)
    rephrased_question = response.content
    state["messages"].append(HumanMessage(content=question, name='user_question'))
    state["question"] = rephrased_question
    state["conversation_summary"] = summary_conversations
    state["memory_index"] = updated_memory_index
    state["messages"].append(AIMessage(content=f"The user provided question has been restructured as: {rephrased_question}", name="restructure_question"))

    print(f"--- Workflow Transition: Rephraser → Supervisor---")

    save_log({
        "user_id": state["user_id"],
        "account_id": state["account_id"],
        "agent_called": "restructure_question",
        "action": "Restructures user query with context",
        "user_query": question,
        "result_summary": f"The user provided question has been restructured as: {rephrased_question}"
    })

    return state

## ----- Supervisor Agent -----

class Supervisor(BaseModel):
    next: Literal["sqlbot", "ragbot", "outofscopequery"] = Field(
        description="Determines which specialist to activate next in the workflow sequence: "
                    "'sqlbot': Handles questions that require accessing transactional data or customer-specific bank records, "
                    "'ragbot': Handles questions related to bank policies, services, FAQs, and general information, "
                    "'outofscopequery': Used when the user's question is not relevant to either SQLBot or RAGBot."
    )
    reason: str = Field(
        description="Detailed justification for the routing decision, explaining the rationale behind selecting the particular specialist or marking the query out-of-scope."
    )

def supervisor_node(state: AgentState) -> Command[Literal["sqlbot", "ragbot", "outofscopequery"]]:

    system_prompt = ('''
        You are a workflow supervisor managing three specialized agents: SQLBot, RAGBot, and an Out-of-Scope handler.
        Your role is to orchestrate the workflow by selecting the most appropriate next agent based on the current state and needs of the task.
        Provide a clear, concise rationale for each decision to ensure transparency in your decision-making process.

        **Agent Descriptions**:

        1. **SQLBot**:
          - Handles questions that require accessing transactional data or customer-specific bank records.
          - Examples:
            - "How much did I spend last month?"
            - "Show my top 5 transactions this week"
            - "Get my account balance"
            - "What was the highest deposit I made last year?"

        2. **RAGBot**:
          - Handles questions related to bank policies, services and FAQs.
          - Examples:
            - "How can I open a new account?"
            - "What’s the minimum balance for a savings account?"
            - "How do I reset my online banking password?"
            - "What are the age requirements for opening a bank account?"

        3. **Out-of-Scope Handler** (`outofscopequery`):
          - Handles user queries that are unrelated to banking transactions or general banking information.
          - Examples:
            - "What is the weather today?"
            - "Tell me a joke"
            - "Who won the cricket match yesterday?"

        **Your Task**:
        Based on the given user question, decide whether the question should be answered by `sqlbot`, `ragbot`, or marked as `outofscopequery`.
        Provide a short, clear explanation of why you chose that agent.
    ''')


    question = state["question"]
    user_prompt = f'''
        User Question: {question}
    '''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = llm.with_structured_output(Supervisor).invoke(messages)

    goto = response.next
    reason = response.reason

    print(f"--- Workflow Transition: Supervisor → {goto.upper()} ---")

    save_log({
        "user_id": state["user_id"],
        "account_id": state["account_id"],
        "agent_called": "supervisor_node",
        "action": "Orchestrates user question to appropriate agent",
        "user_query": state["question"],
        "result_summary": f"Supervisor → {goto.upper()}. Reason: {reason}"
    })

    return Command(
        update={
            "messages": [
                AIMessage(content=reason, name="supervisor")
            ]
        },
        goto=goto,
    )

### --- Out-Of-Scope Handling ---

def outofscopequery(state: AgentState):
    outofscope_response = "This virtual assistant is designed to help with your personal banking transactions and to provide information on bank policies and services. Please ensure your query relates to these areas."
    return Command(
        update={
            "messages": [
                AIMessage(content=outofscope_response, name="outofscopequery")
            ]
        },
        goto=END,
    )

## ----- RAG Node -----

def ragbot(state: AgentState):

    question = state["question"]

    answer = rag_chain(question, state["user_id"], state["account_id"])

    goto = "evaluate_answer"
    print(f"--- Workflow Transition: RAGBot → {goto.upper()} ---")

    save_log({
        "user_id": state["user_id"],
        "account_id": state["account_id"],
        "agent_called": "ragbot",
        "action": "RAGBot Action Completed",
        "user_query": state["question"],
        "result_summary": f"RAGBot → {goto.upper()}."
    })

    return Command(
        update={
            "messages": [
                AIMessage(content=answer, name="ragbot")
            ]
        },
        goto=goto,
    )

## ----- SQL Node -----

def sqlbot(state: AgentState):

      question = state["question"]
      sqlbot_response = sql_graph_manager.invoke({"sql_question": question,
                                          "sql_user_id": state["user_id"],
                                          "sql_account_id": state["account_id"],
                                          "sql_original_question": state["original_question"]})
      answer = sqlbot_response["sql_answer"]

      goto = "evaluate_answer"
      print(f"--- Workflow Transition: SQLBot → {goto.upper()} ---")

      save_log({
          "user_id": state["user_id"],
          "account_id": state["account_id"],
          "agent_called": "sqlbot",
          "action": "SQLBot Action Completed",
          "user_query": state["question"],
          "result_summary": f"SQLBot → {goto.upper()}."
      })

      return Command(
          update={
              "messages": [
                  AIMessage(content=answer, name="sqlbot")
              ]
          },
          goto=goto,
      )

## ----- Evaluate Answer Node -----

class EvaluateAnswer(BaseModel):
    relevance: Literal["Yes", "No"] = Field(
        description="Determines if generated answer is relevant to the user's question: "
                    "Yes: If the generated answer directly address the user's question and seem logically complete"
                    "No: Otherwise "
    )
    justification: str = Field(
        description="Brief justification for the decision, explaining the rationale behind making the particular decision."
    )

def evaluate_answer(state: AgentState):
    """Evaluate the generated answer if relevant to the user's question or not"""
   
    system_prompt = ('''
        You are an evaluator. Your task is to determine if the given answer is relevant, correct, and complete in response to the user's question. 

        You will receive:
        - A user's original natural language question
        - That question restructured by LLM earlier, based on past conversation that you don't have access to
        - An answer generated by the LLM with the agents it has access to:
            1. sqlbot: Handles questions that require accessing transactional data or customer-specific bank records.
            2. ragbot: Handles questions related to bank policies, services, FAQs, and general information.

        Please assess:
        1. Does the answer directly address the user's question?
        2. Is the answer factually correct based on the question?
        3. Does the answer seem logically complete (not too vague, missing, or unrelated)?

        Respond in this format:
        - Relevance: Yes / No
        - Justification: [Brief reason for your decision]
    ''')

    original_user_question = state["original_question"]
    agent_response = state["messages"][-1]

    user_prompt = f'''
        Original User Question: {original_user_question}
        Restructured User Question: {state["question"]}
        Generated Answer: {agent_response}
    '''

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response = llm.with_structured_output(EvaluateAnswer).invoke(messages)

    state["evaluate_answer_status"] = response.relevance

    ea_response = f"Agent Response: {response.relevance}. Reason: {response.justification} "

    save_log({
        "user_id": state["user_id"],
        "account_id": state["account_id"],
        "agent_called": "evaluate_answer",
        "action": "Generated Answer Evaluated",
        "user_query": state["question"],
        "result_summary": ea_response
    })

    return state

def evaluate_answer_router(state: AgentState):

  if state["evaluate_answer_status"] == "Yes":
    save_log({
        "user_id": state["user_id"],
        "account_id": state["account_id"],
        "agent_called": "evaluate_answer_router",
        "action": "Routes to END based on evaluate_answer_status",
        "user_query": state["question"],
        "result_summary": "Moving to END"
    })
    return "END"

  elif state["evaluate_answer_status"] == "No":
    save_log({
        "user_id": state["user_id"],
        "account_id": state["account_id"],
        "agent_called": "evaluate_answer_router",
        "action": "Routes to cannot_answer based on evaluate_answer_status",
        "user_query": state["question"],
        "result_summary": "Moving to cannot_answer"
    })
    return "cannot_answer"

def cannot_answer(state: AgentState):
    "Cannot answer question; return a default answer."
    goto = END
    save_log({
        "user_id": state["user_id"],
        "account_id": state["account_id"],
        "agent_called": "cannot_answer",
        "action": "Cannot answer given question.",
        "user_query": state["question"],
        "result_summary": f"cannot_answer → {goto.upper()}."
    })
  
    cannotanswer_response = "I'm sorry, but I can't answer the given question."
    return Command(
        update={
            "messages": [
                AIMessage(content=cannotanswer_response, name="cannot_answer")
            ]
        },
        goto=goto,
    )


# ----- Defining the Graph -----

from langgraph.checkpoint.memory import MemorySaver

checkpointer = MemorySaver()

graph = StateGraph(AgentState)

graph.add_node("restructure_question", restructure_question)
graph.add_node("supervisor", supervisor_node)
graph.add_node("outofscopequery", outofscopequery)
graph.add_node("ragbot", ragbot)
graph.add_node("sqlbot", sqlbot)
graph.add_node("evaluate_answer", evaluate_answer)
graph.add_node("cannot_answer", cannot_answer)

graph.add_edge(START, "restructure_question")
graph.add_edge("restructure_question", "supervisor")

graph.add_edge("ragbot", "evaluate_answer")
graph.add_edge("sqlbot", "evaluate_answer")
graph.add_conditional_edges(
    "evaluate_answer",
    evaluate_answer_router,
    {
        "END": END,
        "cannot_answer": "cannot_answer",
    }
)

graph.add_edge("outofscopequery", END)
graph.add_edge("cannot_answer", END)

app_manager = graph.compile(checkpointer=checkpointer)
