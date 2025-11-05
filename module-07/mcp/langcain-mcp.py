import os
import asyncio
from dotenv import load_dotenv
from langchain_openai import AzureChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_core.prompts import ChatPromptTemplate

custom_prompt = ChatPromptTemplate.from_template("""
You are a reasoning AI assistant with access to multiple MCP tools:
- Math: for calculations
- Weather: for weather queries
- Microsoft Learn: for documentation search.

Use tools when needed, and clearly show your reasoning steps.
""")


load_dotenv()

async def main():
    # Initialize MCP client with the external Microsoft Learn MCP server
    client = MultiServerMCPClient(
        {
             "math": {
                "command": "python",
                # Use the correct absolute or relative path to your math_server.py
                "args": ["math_server.py"],
                "transport": "stdio",
            },
            "weather": {
                # Ensure your weather MCP server is running here
                "url": "http://localhost:8000/mcp",
                "transport": "streamable_http",
            },
            "learn_docs": {
                "url": "https://learn.microsoft.com/api/mcp",
                "transport": "streamable_http",
            }
            # you can have your local servers too, e.g., math/weather
        }
    )

    tools = await client.get_tools()
    print(f"Loaded tools from learn_docs: {tools}")

    # Initialize Azure model
    deployment_name = "gpt-4o"
    azure_endpoint = "https://ai-new-agent-resource.openai.azure.com/"
    azure_api_key = "3KZOVIIWVSSadEQyVLMr722pVoXVOl6BvatHtqrAsnmSV06FL1SCJQQJ99BIACHYHv6XJ3w3AAAAACOGGKR1"
    azure_api_version = "2024-05-01-preview"

    if not azure_api_key or not azure_endpoint:
        raise ValueError("Missing Azure OpenAI credentials in .env file")

    # Initialize Azure OpenAI model
    model = AzureChatOpenAI(
        azure_endpoint=azure_endpoint,
        api_key=azure_api_key,
        api_version=azure_api_version,
        deployment_name=deployment_name,
    )


    # Create agent with the tools
    agent = create_agent(
        model=model,
        tools=tools
    )

    # Example query
    response = await agent.ainvoke(
        {"messages":[{"role":"user","content":"Search Microsoft docs: how to deploy an Azure Container App"}]}
    )
    print("Response:", response["messages"][-1].content)

if __name__:
    asyncio.run(main())
