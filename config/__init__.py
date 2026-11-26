"""Configuration package for Antigravity local & hybrid agent setups."""

from config.agent_config import (
    DEFAULT_CLOUD_MODEL,
    DEFAULT_EXCLUDED_FOLDERS,
    DEFAULT_LM_STUDIO_BASE_URL,
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_PROJECT_WORKSPACE,
    get_available_local_models,
    get_cloud_agent_config,
    get_configured_workspaces,
    get_lm_studio_models,
    get_local_agent_config,
    resolve_workspaces,
)

__all__ = [
    "DEFAULT_CLOUD_MODEL",
    "DEFAULT_EXCLUDED_FOLDERS",
    "DEFAULT_LM_STUDIO_BASE_URL",
    "DEFAULT_LOCAL_BASE_URL",
    "DEFAULT_LOCAL_MODEL",
    "DEFAULT_PROJECT_WORKSPACE",
    "get_available_local_models",
    "get_cloud_agent_config",
    "get_configured_workspaces",
    "get_lm_studio_models",
    "get_local_agent_config",
    "resolve_workspaces",
]
