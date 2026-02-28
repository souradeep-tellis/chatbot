from langgraph.graph import StateGraph,START,END
from pydantic import BaseModel,Field
from typing import TypedDict,Literal,Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage,HumanMessage,BaseMessage
from dotenv import load_dotenv
import operator
from langgraph.checkpoint.sqlite import SqliteSaver
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_ollama import ChatOllama
import sqlite3

from langgraph.graph import add_messages
class chatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

load_dotenv()
# llm=ChatOpenAI(model="gpt-4o-mini")
# llm=ChatGoogleGenerativeAI(model="gemini-flash-lite-latest")

llm=ChatOllama(
    model=os.environ["OLLAMA_MODEL"],           
    base_url=os.environ["OLLAMA_HOST"],
    validate_model_on_init=True)

def chat_node(state:chatState):
    messages=state['messages']
    
    response=llm.invoke(messages)
    
    return {'messages': [response] }


# here instead of MemorySaver we can also use SqliteSaver or any other saver from langgraph.checkpoint
conn=sqlite3.connect('chatbot.db',check_same_thread=False)

checkpointer=SqliteSaver(conn)
graph=StateGraph(chatState)

graph.add_node("chat_node",chat_node)

graph.add_edge(START,"chat_node")
graph.add_edge("chat_node",END)

chatbot=graph.compile(checkpointer=checkpointer)

def retrieve_all_threads():
    all_threads=set()
    for checkpoint in checkpointer.list(None):
        all_threads.add(checkpoint.config['configurable']['thread_id'])


    return list(all_threads)
