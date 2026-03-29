# agents_ESL

This project demonstrates an **LLM agent that can run engineering simulation tools**.
The current setup exposes a **discrete elastic beam simulation tool**, which the agent can call programmatically.

`main.py` simply runs the **Programmer Agent**, which invokes a tool to run a beam simulation.

## Setup

Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Environment Variable

Create a `.env` file in the project root and add your Anthropic API key:

```text
ANTHROPIC_API_KEY=<your_api_key>
```

The program loads this key using `python-dotenv`.

## Running the Program

Run:

```bash
python main.py
```

`main.py` initializes the **Programmer Agent** and sends it a request to run a beam simulation tool.

Example request executed by the script:

```python
result = programmer_agent.invoke({
    "messages": [
        {"role": "user", "content": "Run discrete_beam with nv=11"}
    ]
})
```

The agent then calls the **Discrete Elastic Beam simulation tool** and prints the result to the terminal.

## Agent Overview

The agent is created using LangChain and is configured to call simulation tools.

The current configuration loads tools from:

```
src.tools.Discrete_Elastic_Beams
```

Only the **discrete beam simulation tool** is exposed to the agent. The agent receives user instructions, runs the tool if needed, and returns the results.
