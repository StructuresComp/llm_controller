from langchain.agents import create_agent
from src.utils.tool_loader import load_tools

# ------ Option 1 ------ #
# Load tool registry from YAML
# tool_registry = load_tools("configs/tools.yaml")

# ------ Option 2 ------ #
# Load tool registry from Python modules
tool_registry = load_tools([
    "src.tools.Discrete_Elastic_Beams",
])

# Select only the beam tool
tools = [tool_registry["discreteBeam"]["func"]]
# Use this version if you want all the tools
# tools = [spec["func"] for spec in tool_registry.values()]

# LLM agent that can call those tools
programmer_agent = create_agent(
    model="claude-sonnet-4-6",
    tools=tools,
    system_prompt=(
        "You are the Programmer agent. "
        "Use the available tools to run simulations or computations "
        "and report the results."
    ),
)