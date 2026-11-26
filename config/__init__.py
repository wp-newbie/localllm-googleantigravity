"""Configuration package for Antigravity local & hybrid agent setups."""

from config.agent_config import (
    DEFAULT_CLOUD_MODEL,
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    get_cloud_agent_config,
    get_local_agent_config,
)

__all__ = [
    "DEFAULT_CLOUD_MODEL",
    "DEFAULT_LOCAL_BASE_URL",
    "DEFAULT_LOCAL_MODEL",
    "get_cloud_agent_config",
    "get_local_agent_config",
]
