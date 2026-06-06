# ---------------------------------------------------------------------------
# src/llm/llm.py
# ---------------------------------------------------------------------------
# Thin adapter over llm_wrapper. Reads provider/model from config.py and
# delegates all provider calls to llm_wrapper.
# ---------------------------------------------------------------------------

from src.core.config import LLM_PROVIDER, MODEL, MAX_TOKENS
from llm_wrapper.llm import call_llm as _call_llm
from llm_wrapper.llm import call_llm_with_tool as _call_llm_with_tool


def call_llm(
    prompt: str,
    system: str = "",
    max_tokens: int = None,
) -> str:
    return _call_llm(
        prompt=prompt,
        model=MODEL,
        provider=LLM_PROVIDER,
        max_tokens=max_tokens or MAX_TOKENS,
        system=system,
    )


def call_llm_with_tool(
    prompt: str,
    tool_schema: dict,
    tool_name: str,
    system: str = "",
    max_tokens: int = None,
) -> dict:
    return _call_llm_with_tool(
        prompt=prompt,
        tool_schema=tool_schema,
        tool_name=tool_name,
        model=MODEL,
        provider=LLM_PROVIDER,
        max_tokens=max_tokens or MAX_TOKENS,
        system=system,
    )
