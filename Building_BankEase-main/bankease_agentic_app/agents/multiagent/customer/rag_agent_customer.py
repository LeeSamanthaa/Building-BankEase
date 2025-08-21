'''
Based on the user question, the top relevant relevant chunks from the vector db is retrieved.
Using the RetrievalQA for retrieval chain.
'''

import os
import json
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA

from dotenv import load_dotenv
load_dotenv()

groq_api_key = os.getenv("GROQ_API_KEY")

llm = ChatGroq(model="llama3-70b-8192", groq_api_key=groq_api_key)

embedding_function = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# ----- Defining Logging -----
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))                # current directory
TRUNK_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..', '..', '..'))      # Move up three levels to reach the trunk
VECTOR_DB_PATH = os.path.join(TRUNK_DIR, 'faiss_database')
LOG_FILE = os.path.join(TRUNK_DIR, 'logs', 'chatbot_logs.jsonl')


def save_log(event: dict):
    os.makedirs("logs", exist_ok=True)
    event["timestamp"] = datetime.now().isoformat()
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(event) + "\n")

# ----- Defining the Retriever -----

vector_store = FAISS.load_local(VECTOR_DB_PATH, embedding_function, allow_dangerous_deserialization=True)
# retriever = vector_store.as_retriever(search_type="mmr", search_kwargs = {"k": 4})
retriever = vector_store.as_retriever()

def rag_chain(question, user_id, account_id):

    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    response = qa_chain.invoke(question)
    answer = response['result']
    
    save_log({
        "user_id": user_id,
        "account_id": account_id,
        "agent_called": "rag_chain",
        "action": "Generating answer with retrieved chunks",
        "user_query": question,
        "result_summary": answer
    })

    return answer