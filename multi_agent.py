"""
multi_agent.py
====================
Two-agent LangGraph loop for refining an NSF proposal abstract.

Graph:
    agent_a (writer) --> agent_b (critic) --> [loop back or END]

State:
    - abstract:   current draft
    - feedback:   latest feedback from Agent B
    - iterations: counter
"""

import os
from typing import TypedDict, Literal
from dotenv import load_dotenv

from langgraph.graph import StateGraph, END
from langchain_anthropic import ChatAnthropic

# ── Setup ──────────────────────────────────────────────────────────────────

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY not found. Add it to .env as ANTHROPIC_API_KEY=<key>.")

llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)

MAX_ITERATIONS = 10
NO_FEEDBACK_SIGNAL = "NO_FEEDBACK"   # sentinel Agent B returns when done


# ── State ──────────────────────────────────────────────────────────────────

class State(TypedDict):
    abstract:   str   # current abstract draft
    feedback:   str   # latest feedback from Agent B
    iterations: int   # how many revision cycles have run


# ── Agent A: Writer ────────────────────────────────────────────────────────

def agent_a(state: State) -> State:
    """Write or revise the abstract based on Agent B's feedback."""

    if not state["abstract"]:
        # First pass — write from scratch
        prompt = (
            "Write a single, concise paragraph abstract for an NSF proposal on "
            "data-driven material modeling. The abstract should highlight the scientific "
            "motivation, the data-driven / machine learning methodology, and the expected "
            "impact on materials science. Write only the abstract paragraph, nothing else."
        )
    else:
        # Revision pass — incorporate feedback
        prompt = (
            f"You are revising an NSF proposal abstract based on reviewer feedback.\n\n"
            f"CURRENT ABSTRACT:\n{state['abstract']}\n\n"
            f"FEEDBACK TO ADDRESS:\n{state['feedback']}\n\n"
            "Revise the abstract to address all feedback points. "
            "Write only the revised abstract paragraph, nothing else."
        )

    response = llm.invoke([{"role": "user", "content": prompt}])
    new_abstract = response.content.strip()

    print(f"\n{'='*60}")
    print(f"[Agent A — Iteration {state['iterations'] + 1}]")
    print(f"{'='*60}")
    print(new_abstract)

    return {**state, "abstract": new_abstract}


# ── Agent B: Critic ────────────────────────────────────────────────────────

def agent_b(state: State) -> State:
    """Review the abstract and return concise feedback, or signal approval."""

    prompt = (
        f"You are a senior NSF reviewer critiquing a proposal abstract.\n\n"
        f"ABSTRACT:\n{state['abstract']}\n\n"
        "Provide concise, specific, actionable feedback to improve this abstract. "
        "Focus on clarity, scientific rigor, methodology, and impact. "
        f"If the abstract is already strong and needs no further improvement, "
        f"reply with exactly the single word: {NO_FEEDBACK_SIGNAL}\n\n"
        "Otherwise, give 2-4 bullet points of specific improvements only."
    )

    response = llm.invoke([{"role": "user", "content": prompt}])
    feedback = response.content.strip()

    print(f"\n[Agent B — Feedback]")
    print("-" * 40)
    print(feedback)

    return {**state, "feedback": feedback, "iterations": state["iterations"] + 1}


# ── Router: decide whether to loop or stop ─────────────────────────────────

def should_continue(state: State) -> Literal["agent_a", "__end__"]:
    """Continue if there is feedback and we haven't hit the iteration limit."""
    if state["iterations"] >= MAX_ITERATIONS:
        print(f"\n[Router] Max iterations ({MAX_ITERATIONS}) reached. Stopping.")
        return "__end__"
    if NO_FEEDBACK_SIGNAL in state["feedback"]:
        print(f"\n[Router] Agent B is satisfied. Stopping.")
        return "__end__"
    return "agent_a"


# ── Build the graph ────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(State)

    graph.add_node("agent_a", agent_a)
    graph.add_node("agent_b", agent_b)

    graph.set_entry_point("agent_a")
    graph.add_edge("agent_a", "agent_b")
    graph.add_conditional_edges("agent_b", should_continue)

    return graph.compile()


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app = build_graph()

    initial_state: State = {
        "abstract":   "",
        "feedback":   "",
        "iterations": 0,
    }

    final_state = app.invoke(initial_state)

    print(f"\n{'='*60}")
    print("FINAL ABSTRACT")
    print(f"{'='*60}")
    print(final_state["abstract"])
    print(f"\nCompleted in {final_state['iterations']} iteration(s).")