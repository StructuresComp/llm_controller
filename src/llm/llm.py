# ---------------------------------------------------------------------------
# src/llm/llm.py
# ---------------------------------------------------------------------------
# Thin adapter over llm_wrapper. Defaults to provider/model from config.py,
# delegates all provider calls to llm_wrapper. Both functions accept optional
# provider/model overrides (e.g. for decision_node's extraction fallback, which
# calls config.EXTRACTION_PROVIDER/EXTRACTION_MODEL instead of the primary one).
#
# When config.LOG_LLM_CALLS is True, every prompt and response is appended
# to logs/<RUN_NAME>.log (or logs/run_<timestamp>_<pid>.log if RUN_NAME is
# unset). Initialization is deferred to the first call so that a --config
# overlay applied in cli.py is visible before the file is opened.
# ---------------------------------------------------------------------------

import os
import time

from src.core import config
from src.core.paths import LOGS_DIR
from llm_wrapper.llm import call_llm as _call_llm
from llm_wrapper.llm import call_llm_with_tool as _call_llm_with_tool

_log_file = None
_initialized = False


def _init() -> None:
    global _log_file, _initialized
    if _initialized:
        return
    _initialized = True

    # Push Ollama settings into env so llm_wrapper picks them up.
    if config.LLM_PROVIDER == "ollama":
        os.environ.setdefault("OLLAMA_BASE_URL",    config.OLLAMA_BASE_URL)
        os.environ.setdefault("OLLAMA_TIMEOUT",     str(config.OLLAMA_TIMEOUT))
        os.environ.setdefault("OLLAMA_TEMPERATURE", str(config.OLLAMA_TEMPERATURE))

    if config.LOG_LLM_CALLS:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        stem = getattr(config, "RUN_NAME", "") or f"run_{int(time.time())}_{os.getpid()}"
        _log_file = open(LOGS_DIR / f"{stem}.log", "w", buffering=1)  # line-buffered


def _log(role: str, content: str) -> None:
    if _log_file:
        _log_file.write(f"\n{'='*60}\n[{role}]\n{content}\n")


def call_llm(
    prompt: str,
    system: str = "",
    max_tokens: int = None,
    provider: str = None,
    model: str = None,
) -> str:
    """provider/model override config.LLM_PROVIDER/config.MODEL when given —
    used by decision_node's extraction fallback to call a different model."""
    _init()
    _log("PROMPT", f"system: {system}\n---\n{prompt}")
    response = _call_llm(
        prompt=prompt,
        model=model or config.MODEL,
        provider=provider or config.LLM_PROVIDER,
        max_tokens=max_tokens or config.MAX_TOKENS,
        system=system,
    )
    _log("RESPONSE", response)
    return response


def call_llm_with_tool(
    prompt: str,
    tool_schema: dict,
    tool_name: str,
    system: str = "",
    max_tokens: int = None,
    provider: str = None,
    model: str = None,
) -> dict:
    """provider/model override config.LLM_PROVIDER/config.MODEL when given —
    used by decision_node's extraction fallback to call a different model."""
    _init()
    _log("PROMPT (tool)", f"tool: {tool_name}\nsystem: {system}\n---\n{prompt}")
    response = _call_llm_with_tool(
        prompt=prompt,
        tool_schema=tool_schema,
        tool_name=tool_name,
        model=model or config.MODEL,
        provider=provider or config.LLM_PROVIDER,
        max_tokens=max_tokens or config.MAX_TOKENS,
        system=system,
    )
    _log("RESPONSE (tool)", str(response))
    return response
