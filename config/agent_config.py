"""Agent configuration module for Google Antigravity SDK.

Provides modular configurations for local inference (Ollama, LM Studio, vLLM)
using `LocalOpenAIAgentConfig` with `.lightweight()` optimizations, as well
as cloud agent configurations for high-level reasoning and planning.
Includes flexible workspace inclusion, discovery, and folder exclusion controls.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Optional
import requests
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

DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434/v1"
DEFAULT_LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

DEFAULT_LOCAL_BASE_URL = DEFAULT_OLLAMA_BASE_URL
DEFAULT_LOCAL_MODEL = "qwen2.5-coder:7b"
DEFAULT_CLOUD_MODEL = "gemini-2.5-flash"

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


def get_lm_studio_models(base_url: str = DEFAULT_LM_STUDIO_BASE_URL) -> list[str]:
    """Queries LM Studio for loaded models and scans local disk for downloaded GGUFs."""
    models: list[str] = []
    seen: set[str] = set()

    # 1. Query running LM Studio OpenAI endpoint (/v1/models)
    models_url = f"{base_url.rstrip('/')}/models"
    try:
        resp = requests.get(models_url, timeout=1.5)
        if resp.status_code == 200:
            data = resp.json().get("data", [])
            for item in data:
                mid = item.get("id")
                # Filter out embedding-only models from primary LLM selection
                if mid and "embed" not in mid.lower():
                    if mid.lower() not in seen:
                        seen.add(mid.lower())
                        models.append(mid)
    except Exception:
        pass

    # 2. Scan LM Studio disk directory (~/.lmstudio/models)
    lm_dir = os.path.expanduser("~/.lmstudio/models")
    if os.path.isdir(lm_dir):
        try:
            for root, _, files in os.walk(lm_dir):
                for f in files:
                    if f.endswith(".gguf") and "embed" not in f.lower():
                        name = f.replace(".gguf", "")
                        if name.lower() not in seen:
                            seen.add(name.lower())
                            models.append(name)
        except Exception:
            pass

    return models if models else ["qwen2.5-7b-instruct-1m"]


def get_ollama_models(base_url: str = DEFAULT_OLLAMA_BASE_URL) -> list[str]:
    """Queries Ollama for locally downloaded and tagged models."""
    models: list[str] = []
    seen: set[str] = set()

    # Try /v1/models then /api/tags
    v1_url = f"{base_url.rstrip('/')}/models"
    native_url = f"{base_url.replace('/v1', '').rstrip('/')}/api/tags"

    for url in [v1_url, native_url]:
        try:
            resp = requests.get(url, timeout=1.5)
            if resp.status_code == 200:
                data = resp.json()
                if "data" in data and isinstance(data["data"], list):
                    for item in data["data"]:
                        mid = item.get("id")
                        if mid and mid.lower() not in seen:
                            seen.add(mid.lower())
                            models.append(mid)
                elif "models" in data and isinstance(data["models"], list):
                    for item in data["models"]:
                        mid = item.get("name")
                        if mid and mid.lower() not in seen:
                            seen.add(mid.lower())
                            models.append(mid)
            if models:
                break
        except Exception:
            continue

    return models if models else ["qwen2.5-coder:7b", "gemma4:latest", "qwen2.5-coder:0.5b"]


def get_available_local_models(provider: str = "auto", base_url: Optional[str] = None) -> list[str]:
    """Discovers available models based on provider (lmstudio, ollama, or auto)."""
    prov = provider.lower()
    if "lm" in prov:
        return get_lm_studio_models(base_url or DEFAULT_LM_STUDIO_BASE_URL)
    elif "ollama" in prov:
        return get_ollama_models(base_url or DEFAULT_OLLAMA_BASE_URL)

    # Auto: Check base_url or query both
    if base_url:
        if "1234" in base_url:
            return get_lm_studio_models(base_url)
        elif "11434" in base_url:
            return get_ollama_models(base_url)

    # Return union of both
    lm = get_lm_studio_models()
    ol = get_ollama_models()
    combined: list[str] = []
    seen: set[str] = set()
    for m in lm + ol:
        if m.lower() not in seen:
            seen.add(m.lower())
            combined.append(m)
    return combined


def get_configured_workspaces(project_dir: Optional[str] = None) -> list[str]:
    """Discovers all workspace directories defined via ANTIGRAVITY_WORKSPACE(S)."""
    discovered: list[str] = []
    seen: set[str] = set()

    def add_path(p: str):
        cleaned = p.strip().strip("'\"").rstrip("/\\")
        if cleaned:
            normalized = os.path.normpath(cleaned)
            if normalized.lower() not in seen:
                seen.add(normalized.lower())
                discovered.append(normalized)

    # 1. Environment variables
    for env_var in ["ANTIGRAVITY_WORKSPACE", "ANTIGRAVITY_WORKSPACES"]:
        val = os.getenv(env_var)
        if val:
            for part in re.split(r"[,;]+", val):
                add_path(part)

    # 2. Check .env and .env.example files
    base_dirs = [
        project_dir,
        os.getcwd(),
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    ]
    for bdir in base_dirs:
        if not bdir:
            continue
        for fname in [".env", ".env.local", ".env.example"]:
            fpath = os.path.join(bdir, fname)
            if os.path.isfile(fpath):
                try:
                    with open(fpath, "r", encoding="utf-8") as f:
                        for line in f:
                            line = line.strip()
                            if line.startswith("#") or not line:
                                continue
                            m = re.match(r"^ANTIGRAVITY_WORKSPACE(?:S)?\s*=\s*(.+)$", line, re.IGNORECASE)
                            if m:
                                raw_val = m.group(1).strip()
                                for part in re.split(r"[,;]+", raw_val):
                                    add_path(part)
                except Exception:
                    pass

    # 3. Add default candidates if present on disk
    for default_candidate in ["F:\\Personal", "F:\\Code"]:
        if os.path.isdir(default_candidate):
            add_path(default_candidate)

    return discovered


_ALL_DISCOVERED = get_configured_workspaces()
DEFAULT_PROJECT_WORKSPACE = _ALL_DISCOVERED[0] if _ALL_DISCOVERED else None


def resolve_workspaces(
    workspaces: Optional[list[str] | str] = None,
    enabled: bool = True,
) -> Optional[list[str]]:
    """Resolves target project workspace directories for the agent."""
    if not enabled:
        return None

    if workspaces == "" or workspaces == []:
        return None

    if isinstance(workspaces, str):
        trimmed = workspaces.strip()
        if not trimmed:
            return None
        workspaces = [trimmed]

    if workspaces:
        valid_dirs: list[str] = []
        for p in workspaces:
            if isinstance(p, str) and p.strip() and os.path.isdir(p.strip()):
                norm = os.path.abspath(p.strip())
                if norm not in valid_dirs:
                    valid_dirs.append(norm)
        return valid_dirs if valid_dirs else None

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
