import time
from agents.multiagent.multi_agent import app 

def make_output(query, user_id, account_id, thread):
    answer = app.invoke({"question": query, 
                "user_id": user_id, "account_id": account_id},
                config={"configurable": {"thread_id": thread}})
    result = answer["messages"][-1].content
    return result 

def modify_output(input):
    for text in input.split():
        yield text + " "
        time.sleep(0.05)