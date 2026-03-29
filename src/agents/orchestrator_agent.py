from langchain.agents import create_agent
from src.agents.programmer_tool import run_beam_simulation

orchestrator_agent = create_agent(
    model="claude-sonnet-4-6",
    tools=[run_beam_simulation],
    system_prompt=(
        "You are the Orchestrator agent.\n"
        "Goal: perform mesh sensitivity analysis.\n"
        "Find the smallest nv that produces a stable beam solution.\n"
        "nv must be between 3 and 30.\n"
        "Use run_beam_simulation to test values of nv.\n"
        "Prefer smaller nv for computational efficiency."
    ),
)