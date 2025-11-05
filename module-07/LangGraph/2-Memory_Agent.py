import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv



import os 
load_dotenv()

# Retrieve credentials from environment
azure_endpoint = "https://ai-new-agent-resource.openai.azure.com/"
azure_api_key = "3KZOVIIWVSSadEQyVLMr722pVoXVOl6BvatHtqrAsnmSV06FL1SCJQQJ99BIACHYHv6XJ3w3AAAAACOGGKR1"
azure_api_version = "2024-05-01-preview"


if not azure_api_key or not azure_endpoint:
    raise ValueError(
        "Missing Azure OpenAI credentials. Please set AZURE_OPENAI_API_KEY and AZURE_OPENAI_ENDPOINT."
    )


# Initialize Azure OpenAI model
llm = AzureChatOpenAI(
    azure_endpoint=azure_endpoint,
    api_key=azure_api_key,
    api_version=azure_api_version,
    deployment_name="gpt-4o"  # must match your Azure deployment name
)

class AgentState(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]



def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["messages"])

    state["messages"].append(AIMessage(content=response.content)) 
    print(f"\nAI: {response.content}")
    print("CURRENT STATE: ", state["messages"])

    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()


conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": conversation_history})
    conversation_history = result["messages"]
    user_input = input("Enter: ")


with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End of Conversation")

print("Conversation saved to logging.txt")