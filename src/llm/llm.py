# ---------------------------------------------------------------------------
# src/llm/llm.py
# ---------------------------------------------------------------------------
# Single entry point for all LLM calls.
#
# Agents call call_llm(..., model="fast"/"smart") and the framework routes
# to the correct provider. get_langchain_llm() returns a LangChain-compatible
# chat model for use inside LangGraph nodes.
#
# All calls are logged to LOGS_DIR/llm_calls.jsonl for debugging and analysis.
# ---------------------------------------------------------------------------

import json
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

from config import (
    LLM_PROVIDER_FAST,
    FAST_MODEL,
    LLM_PROVIDER_SMART,
    SMART_MODEL,
    LOG_LLM_CALLS,
)
from paths import LOGS_DIR
from src.llm.providers.base_provider import BaseProvider

load_dotenv()

# Ensure logs directory exists
LOGS_DIR.mkdir(parents=True, exist_ok=True)


def _make_provider(name: str) -> BaseProvider:
    if name == "anthropic":
        from src.llm.providers.anthropic_provider import AnthropicProvider
        return AnthropicProvider()
    if name == "openai":
        from src.llm.providers.openai_provider import OpenAIProvider
        return OpenAIProvider()
    if name == "ollama":
        from src.llm.providers.ollama_provider import OllamaProvider
        return OllamaProvider()
    if name == "fake":
        from src.llm.providers.fake_provider import FakeProvider
        return FakeProvider()
    raise ValueError(f"Unknown provider: {name}")


fast_provider: BaseProvider = _make_provider(LLM_PROVIDER_FAST)
smart_provider: BaseProvider = _make_provider(LLM_PROVIDER_SMART)


def _get_provider(model_tier: str) -> BaseProvider:
    if model_tier == "fast":
        return fast_provider
    if model_tier == "smart":
        return smart_provider
    return fast_provider  # full model-name passthrough: route via fast provider


def _resolve_model(model: str) -> str:
    if model == "fast":
        return FAST_MODEL
    if model == "smart":
        return SMART_MODEL
    return model


def _log_call(call_type: str, model: str, system: str, prompt: str,
              response: str, tool_name: str = None) -> None:
    """Log an LLM call to LOGS_DIR/llm_calls.jsonl (JSON lines format) if LOG_LLM_CALLS is True."""
    if not LOG_LLM_CALLS:
        return

    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "call_type": call_type,  # "text" or "tool"
        "model": model,
        "system": system,
        "prompt": prompt,
        "response": response,
    }
    if tool_name:
        log_entry["tool_name"] = tool_name

    log_file = LOGS_DIR / "llm_calls.jsonl"
    with open(log_file, "a") as f:
        f.write(json.dumps(log_entry) + "\n")


def call_llm(
    prompt: str,
    max_tokens: int = 4096,
    model: str = "fast",
    system: str = "",
) -> str:
    provider = _get_provider(model)
    resolved_model = _resolve_model(model)
    response = provider.call_text(
        prompt=prompt,
        model=resolved_model,
        max_tokens=max_tokens,
        system=system,
    )
    _log_call(call_type="text", model=resolved_model, system=system,
              prompt=prompt, response=response)
    return response


def call_llm_with_tool(
    prompt: str,
    tool_schema: dict,
    tool_name: str,
    max_tokens: int = 4096,
    model: str = "fast",
    system: str = "",
) -> dict:
    provider = _get_provider(model)
    resolved_model = _resolve_model(model)
    response = provider.call_tool(
        prompt=prompt,
        tool_schema=tool_schema,
        tool_name=tool_name,
        model=resolved_model,
        max_tokens=max_tokens,
        system=system,
    )
    _log_call(call_type="tool", model=resolved_model, system=system,
              prompt=prompt, response=json.dumps(response), tool_name=tool_name)
    return response


def get_langchain_llm(tier: str = "fast"):
    """Return a LangChain chat model driven by config.py provider settings."""
    from config import OLLAMA_BASE_URL
    provider = LLM_PROVIDER_FAST if tier == "fast" else LLM_PROVIDER_SMART
    model    = FAST_MODEL        if tier == "fast" else SMART_MODEL

    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        return ChatAnthropic(model=model)
    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(model=model)
    if provider == "ollama":
        from langchain_ollama import ChatOllama
        return ChatOllama(model=model, base_url=OLLAMA_BASE_URL)

    raise ValueError(f"No LangChain wrapper for provider: {provider}")
