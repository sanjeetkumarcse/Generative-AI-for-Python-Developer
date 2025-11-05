import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI

async def main():
    # Initialize the MCP client with both servers
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
        }
    )

    # Fetch available tools from both MCP servers
    tools = await client.get_tools()

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


    # Create a ReAct agent using your desired model + tools
    agent = create_agent(model, tools)

    # Run queries through the agent
    math_response = await agent.ainvoke({"messages": [{"role": "user", "content": "what's (3 + 5) x 12?"}]})
    print("Math Response:", math_response["messages"][-1].content)

    weather_response = await agent.ainvoke({"messages": [{"role": "user", "content": "what is the weather in nyc?"}]})
    print("Weather Response:", weather_response["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
