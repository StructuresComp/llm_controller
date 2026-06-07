# ---------------------------------------------------------------------------
# src/llm/llm.py
# ---------------------------------------------------------------------------
# Thin adapter over llm_wrapper. Reads provider/model from config.py and
# delegates all provider calls to llm_wrapper.
# ---------------------------------------------------------------------------

import os

from src.core import config
from llm_wrapper.llm import call_llm as _call_llm
from llm_wrapper.llm import call_llm_with_tool as _call_llm_with_tool

# Push Ollama settings from config into env so llm_wrapper picks them up.
# llm_wrapper reads these from os.environ; config.py is the source of truth.
if config.LLM_PROVIDER == "ollama":
    os.environ.setdefault("OLLAMA_BASE_URL",    config.OLLAMA_BASE_URL)
    os.environ.setdefault("OLLAMA_TIMEOUT",     str(config.OLLAMA_TIMEOUT))
    os.environ.setdefault("OLLAMA_TEMPERATURE", str(config.OLLAMA_TEMPERATURE))


def call_llm(
    prompt: str,
    system: str = "",
    max_tokens: int = None,
) -> str:
    return _call_llm(
        prompt=prompt,
        model=config.MODEL,
        provider=config.LLM_PROVIDER,
        max_tokens=max_tokens or config.MAX_TOKENS,
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
        model=config.MODEL,
        provider=config.LLM_PROVIDER,
        max_tokens=max_tokens or config.MAX_TOKENS,
        system=system,
    )
