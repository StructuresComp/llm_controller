import os
from dotenv import load_dotenv
from langchain.agents import create_agent

load_dotenv()  # reads .env in cwd

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY not found. Put it in .env as ANTHROPIC_API_KEY=<key>.")

def get_weather(city: str) -> str:
    """Get weather for a given city."""
    return f"It is sunny in {city}."

agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[get_weather],
    system_prompt="You are a helpful assistant.",
)

result = agent.invoke({"messages": [{"role": "user", "content": "What is the weather in San Diego?"}]})
# print(result)
print(result["messages"][-1].content)