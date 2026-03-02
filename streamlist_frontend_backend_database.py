import streamlit as st
from langchain_core.messages import HumanMessage
import uuid

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
import os



from langgraph.graph import add_messages
class chatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]

model_url=os.getenv("X_URL")
load_dotenv()

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




# ***********************Utility function **************************
def generate_thred_id():
    thread = str(uuid.uuid4())
    return thread


def reset_chat():
    thread_id = generate_thred_id()
    st.session_state["thread_id"] = thread_id
    add_thread(st.session_state["thread_id"])
    st.session_state["message_history"] = []


def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state["chat_threads"].append(thread_id)
        #add default title to the thread
        st.session_state["chat_titles"][thread_id] = "New Chat"


def load_conversation(thread_id):
    state = chatbot.get_state(config={'configurable': {'thread_id': thread_id}})
    # Check if messages key exists in state values, return empty list if not
    return state.values.get('messages', []) 

def get_chat_title(thread_id):
    #if a title already esists return it
    title=st.session_state["chat_titles"].get(thread_id,"New Chat")
    if title!="New Chat":
        return title
    
    #if it's still New chat see of there are any conversation, then send those messages to ai and ask for a title
    messages=load_conversation(thread_id)
    if messages:
        llm_response=llm.invoke(messages+[HumanMessage(content="Based on the conversation above, give me a short and concise title for this chat thread. Reply with just the title and nothing else.")])
        title=llm_response.content.strip()
        st.session_state["chat_titles"][thread_id]=title
        return title
    
    return "New Chat"




# **********************Session management **************************
if "message_history" not in st.session_state:
    st.session_state["message_history"] = []

if "thread_id" not in st.session_state:
    st.session_state["thread_id"] = generate_thred_id()

if "chat_threads" not in st.session_state:
    st.session_state["chat_threads"] = retrieve_all_threads()

if "chat_titles" not in st.session_state:
    st.session_state["chat_titles"] = {}

add_thread(st.session_state['thread_id'])


# *********************Sidebar UI **************************
st.sidebar.title("Langgraph chatbot")

if st.sidebar.button('New Chat'):
    reset_chat()

st.sidebar.header("My conversations")

for thread_id in st.session_state["chat_threads"][::-1]:
    chat_title=get_chat_title(thread_id)
    if st.sidebar.button(chat_title,key=thread_id):
        st.session_state['thread_id'] = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []

        for msg in messages:
            if isinstance(msg, HumanMessage):
                role='user'
            else:
                role='assistant'
            temp_messages.append({'role': role, 'content': msg.content})

        st.session_state['message_history'] = temp_messages





# *****************************Main UI **************************

# showing conversation history
for message in st.session_state["message_history"]:
    with st.chat_message(message["role"]):
        st.text(message["content"])



user_input = st.chat_input("Type your message here")

if user_input:

    if len(st.session_state["message_history"]) == 0:
        title = user_input[:30] + "..." if len(user_input) > 30 else user_input
        st.session_state["chat_titles"][st.session_state["thread_id"]] = title
        st.sidebar.empty() # Optional: slight trick to force sidebar to update if needed

    st.session_state["message_history"].append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.text(user_input)


    CONFIG = {"configurable": {"thread_id": st.session_state["thread_id"]}}

    # invoke api
    # state={'messages':[HumanMessage(content=user_input)]}
    # response=chatbot.invoke(state,config=CONFIG)
    # ai_response=response['messages'][-1].content
    # st.session_state['message_history'].append({"role":"assistant","content":ai_response})
    with st.chat_message("assistant"):
        ai_message = st.write_stream(
            message_chunk.content
            for message_chunk, metadata in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=CONFIG,
                stream_mode="messages",
            )
        )

    st.session_state["message_history"].append(
        {"role": "assistant", "content": ai_message}
    )

    if len( st.session_state["message_history"]) >1:
        st.rerun()



