"""Agent configuration module for Google Antigravity SDK.

Provides modular configurations for local inference (Ollama, LM Studio, vLLM)
using `LocalOpenAIAgentConfig` with `.lightweight()` optimizations, as well
as cloud agent configurations for high-level reasoning and planning.
Includes flexible workspace inclusion and folder exclusion controls.
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

DEFAULT_EXCLUDED_FOLDERS = [
    "node_modules",
    "dist",
    "release",
    ".git",
    ".venv",
    "build",
    "__pycache__",
]

DEFAULT_LOCAL_SYSTEM_INSTRUCTIONS = (
    "You are an efficient, accurate local coding assistant running on a local LLM. "
    "Keep responses concise, prioritize direct code answers, and minimize unnecessary token overhead. "
    "When project workspaces are attached, inspect and reference local files when answering questions."
)

DEFAULT_CLOUD_SYSTEM_INSTRUCTIONS = (
    "You are an expert software architect and strategic planner. "
    "Provide deep technical reasoning, thorough system architecture, and robust planning."
)


def resolve_workspaces(
    workspaces: Optional[list[str] | str] = None,
    enabled: bool = True,
) -> Optional[list[str]]:
    """Resolves target project workspace directories for the agent.

    Args:
        workspaces: Directory path(s), empty string/list, or None.
        enabled: If False, workspace is explicitly excluded (returns None).

    Returns:
        List of absolute valid directory paths, or None if excluded/empty.
    """
    if not enabled:
        return None

    # Empty string or empty list means user explicitly cleared/excluded workspace
    if workspaces == "" or workspaces == []:
        return None

    if isinstance(workspaces, str):
        trimmed = workspaces.strip()
        if not trimmed:
            return None
        workspaces = [trimmed]

    if workspaces:
        valid_dirs = [os.path.abspath(p) for p in workspaces if os.path.isdir(p)]
        return valid_dirs if valid_dirs else None

    # Only fall back to DEFAULT_PROJECT_WORKSPACE when workspaces is strictly None (unspecified)
    if workspaces is None and DEFAULT_PROJECT_WORKSPACE and os.path.isdir(DEFAULT_PROJECT_WORKSPACE):
        return [os.path.abspath(DEFAULT_PROJECT_WORKSPACE)]

    return None


def get_local_agent_config(
    model: str = DEFAULT_LOCAL_MODEL,
    base_url: str = DEFAULT_LOCAL_BASE_URL,
    lightweight: bool = True,
    system_instructions: Optional[str] = None,
    workspaces: Optional[list[str] | str] = None,
    include_workspace: bool = True,
    exclude_folders: Optional[list[str]] = None,
    **kwargs: Any,
) -> LocalOpenAIAgentConfig:
    """Creates a LocalOpenAIAgentConfig tuned for local OpenAI-compatible endpoints."""
    instructions = system_instructions or DEFAULT_LOCAL_SYSTEM_INSTRUCTIONS
    active_workspaces = resolve_workspaces(workspaces, enabled=include_workspace)

    # Append folder exclusions to instructions so the agent avoids heavy subdirectories
    exclusions = exclude_folders if exclude_folders is not None else DEFAULT_EXCLUDED_FOLDERS
    if active_workspaces and exclusions:
        instructions += f"\nStrictly ignore and do not traverse or read files within these excluded directories: {', '.join(exclusions)}."

    config = LocalOpenAIAgentConfig(
        model=model,
        base_url=base_url,
        system_instructions=instructions,
        workspaces=active_workspaces,
        **kwargs,
    )

    if lightweight:
        logger.info(
            "Enabling .lightweight() preset for model '%s' at %s (workspaces: %s, excluded: %s)",
            model,
            base_url,
            active_workspaces,
            exclusions,
        )
        config = config.lightweight()

    return config


def get_cloud_agent_config(
    model: str = DEFAULT_CLOUD_MODEL,
    system_instructions: Optional[str] = None,
    workspaces: Optional[list[str] | str] = None,
    include_workspace: bool = True,
    exclude_folders: Optional[list[str]] = None,
    **kwargs: Any,
) -> LocalAgentConfig:
    """Creates a LocalAgentConfig for cloud models (e.g. Gemini) for complex planning tasks."""
    instructions = system_instructions or DEFAULT_CLOUD_SYSTEM_INSTRUCTIONS
    active_workspaces = resolve_workspaces(workspaces, enabled=include_workspace)

    exclusions = exclude_folders if exclude_folders is not None else DEFAULT_EXCLUDED_FOLDERS
    if active_workspaces and exclusions:
        instructions += f"\nStrictly ignore and do not traverse or read files within these excluded directories: {', '.join(exclusions)}."

    config = LocalAgentConfig(
        model=model,
        system_instructions=instructions,
        workspaces=active_workspaces,
        **kwargs,
    )
    return config
