from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_openai import AzureChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # used to store secret stuff like API keys or configuration values


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
    messages: List[HumanMessage]


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END) 
agent = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"messages": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")





