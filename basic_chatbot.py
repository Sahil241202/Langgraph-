import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from typing import Annotated
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages

# Load API keys from .env
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")
langsmith = os.getenv("LANGSMITH_API_KEY")

# Set environment variables
os.environ["LANGCHAIN_API_KEY"] = langsmith
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "CourseLanggraph"

# Initialize LLM
llm = ChatGroq(groq_api_key=groq_api_key, model_name="Gemma2-9b-It")

class State(TypedDict):
    messages: Annotated[list, add_messages]  # Stores chat messages

# Create Langraph state graph
graph_builder = StateGraph(State)

def chatbot(state: State):
    return {"messages": llm.invoke(state["messages"])}

graph_builder.add_node("chatbot", chatbot)
graph_builder.add_edge(START, "chatbot")
graph_builder.add_edge("chatbot", END)

graph = graph_builder.compile()

# Chat loop
while True:
    user_input = input("User: ")
    if user_input.lower() in ["quit", "q"]:
        print("Good Bye")
        break
    for event in graph.stream({"messages": ("user", user_input)}):
        for value in event.values():
            print("Assistant:", value["messages"].content)
