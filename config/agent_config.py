"""Agent configuration module for Google Antigravity SDK.

Provides modular configurations for local inference (Ollama, LM Studio, vLLM)
using `LocalOpenAIAgentConfig` with `.lightweight()` optimizations, as well
as cloud agent configurations for high-level reasoning and planning.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional
from google.antigravity import (
    Agent,
    AgentBehavior,
    BuiltinTools,
    CapabilitiesConfig,
    CompactionConfig,
    LocalAgentConfig,
    LocalOpenAIAgentConfig,
)

logger = logging.getLogger(__name__)

DEFAULT_LOCAL_MODEL = "qwen2.5-coder:7b"
DEFAULT_LOCAL_BASE_URL = "http://localhost:11434/v1"
DEFAULT_CLOUD_MODEL = "gemini-2.5-flash"

# Automatically resolve default project workspace folder
DEFAULT_PROJECT_WORKSPACE = os.getenv(
    "ANTIGRAVITY_WORKSPACE",
    "F:\\Personal" if os.path.exists("F:\\Personal") else None,
)

DEFAULT_LOCAL_SYSTEM_INSTRUCTIONS = (
    "You are an efficient, accurate local coding assistant running on a local LLM. "
    "Keep responses concise, prioritize direct code answers, and minimize unnecessary token overhead. "
    "When project workspaces are attached, inspect and reference local files when answering questions."
)

DEFAULT_CLOUD_SYSTEM_INSTRUCTIONS = (
    "You are an expert software architect and strategic planner. "
    "Provide deep technical reasoning, thorough system architecture, and robust planning."
)


def resolve_workspaces(workspaces: Optional[list[str] | str] = None) -> Optional[list[str]]:
    """Resolves target project workspace directories for the agent."""
    if isinstance(workspaces, str):
        workspaces = [workspaces]

    if workspaces:
        valid_dirs = [os.path.abspath(p) for p in workspaces if os.path.isdir(p)]
        if valid_dirs:
            return valid_dirs

    if DEFAULT_PROJECT_WORKSPACE and os.path.isdir(DEFAULT_PROJECT_WORKSPACE):
        return [os.path.abspath(DEFAULT_PROJECT_WORKSPACE)]

    return None


def get_local_agent_config(
    model: str = DEFAULT_LOCAL_MODEL,
    base_url: str = DEFAULT_LOCAL_BASE_URL,
    lightweight: bool = True,
    system_instructions: Optional[str] = None,
    workspaces: Optional[list[str] | str] = None,
    **kwargs: Any,
) -> LocalOpenAIAgentConfig:
    """Creates a LocalOpenAIAgentConfig tuned for local OpenAI-compatible endpoints.

    Args:
        model: Model identifier in Ollama (e.g. 'qwen2.5-coder:7b' or 'gemma4').
        base_url: OpenAI-compatible API base URL (default 'http://localhost:11434/v1').
        lightweight: Whether to enable .lightweight() optimization for local inference.
        system_instructions: Optional custom system instructions.
        workspaces: Target project directory or list of directories (e.g. ['F:\\Personal']).
        **kwargs: Additional parameters forwarded to LocalOpenAIAgentConfig.

    Returns:
        Configured LocalOpenAIAgentConfig instance.
    """
    instructions = system_instructions or DEFAULT_LOCAL_SYSTEM_INSTRUCTIONS
    active_workspaces = resolve_workspaces(workspaces)

    config = LocalOpenAIAgentConfig(
        model=model,
        base_url=base_url,
        system_instructions=instructions,
        workspaces=active_workspaces,
        **kwargs,
    )

    if lightweight:
        logger.info(
            "Enabling .lightweight() preset for model '%s' at %s (workspaces: %s)",
            model,
            base_url,
            active_workspaces,
        )
        config = config.lightweight()

    return config


def get_cloud_agent_config(
    model: str = DEFAULT_CLOUD_MODEL,
    system_instructions: Optional[str] = None,
    workspaces: Optional[list[str] | str] = None,
    **kwargs: Any,
) -> LocalAgentConfig:
    """Creates a LocalAgentConfig for cloud models (e.g. Gemini) for complex planning tasks.

    Args:
        model: Frontier cloud model target (e.g. 'gemini-2.5-pro' or 'gemini-2.5-flash').
        system_instructions: Strategic planning & reasoning instructions.
        workspaces: Target project directory or list of directories.
        **kwargs: Additional parameters forwarded to LocalAgentConfig.

    Returns:
        Configured LocalAgentConfig instance.
    """
    instructions = system_instructions or DEFAULT_CLOUD_SYSTEM_INSTRUCTIONS
    active_workspaces = resolve_workspaces(workspaces)

    config = LocalAgentConfig(
        model=model,
        system_instructions=instructions,
        workspaces=active_workspaces,
        **kwargs,
    )
    return config
