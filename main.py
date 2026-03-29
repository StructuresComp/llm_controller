import os
from dotenv import load_dotenv

load_dotenv()  # loads .env into environment

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY not found. Put it in .env as ANTHROPIC_API_KEY=<key>.")

from src.agents.programmer_agent import programmer_agent

result = programmer_agent.invoke({
    "messages": [
        {"role": "user", "content": "Run discrete_beam with nv=11"}
    ]
})

print(result["messages"][-1].content)