# ---------------------------------------------------------------------------
# agentic_writing/core/providers/base_provider.py
# ---------------------------------------------------------------------------
# Abstract interface that all LLM providers must implement.
# Defines the two call types used across the framework: plain text and
# forced tool use. Each provider also declares its fast/smart model IDs
# so llm.py can expose provider-agnostic MODEL_FAST / MODEL_SMART constants.
# ---------------------------------------------------------------------------

from abc import ABC, abstractmethod


class BaseProvider(ABC):
    fast_model: str
    smart_model: str

    @abstractmethod
    def call_text(
        self,
        prompt: str,
        model: str,
        max_tokens: int,
        system: str,
    ) -> str: ...

    @abstractmethod
    def call_tool(
        self,
        prompt: str,
        tool_schema: dict,
        tool_name: str,
        model: str,
        max_tokens: int,
        system: str,
    ) -> dict: ...
