"""Hybrid Router for Google Antigravity SDK.

Coordinates execution between compact local models (via LocalOpenAIAgentConfig)
and frontier cloud models (via LocalAgentConfig). Intelligently routes tasks
based on complexity, local availability, and provides automatic fallback
when local endpoints are degraded or overloaded.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from enum import Enum
from typing import AsyncGenerator, Optional, Tuple

import requests
from google.antigravity import Agent
from config.agent_config import (
    DEFAULT_CLOUD_MODEL,
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    get_cloud_agent_config,
    get_local_agent_config,
)

logger = logging.getLogger(__name__)


class TaskComplexity(str, Enum):
    """Classification of agent task requirements."""
    LOCAL_LIGHTWEIGHT = "LOCAL_LIGHTWEIGHT"  # Code completions, snippets, quick edits, local fixes
    CLOUD_PLANNING = "CLOUD_PLANNING"        # Architecture, large refactors, multi-step system design


@dataclass
class RouteDecision:
    target: str  # "local" or "cloud"
    complexity: TaskComplexity
    reason: str


class HybridAgentRouter:
    """Intelligent router managing local inference and cloud escalation."""

    PLANNING_KEYWORDS = [
        r"\barchitecture\b",
        r"\bsystem\s+design\b",
        r"\broadmap\b",
        r"\bstrategic\b",
        r"\bmulti[- ]phase\b",
        r"\bcomprehensive\s+plan\b",
        r"\bmigration\s+plan\b",
        r"\bdatabase\s+schema\s+design\b",
        r"\bhigh[- ]level\s+design\b",
        r"\bsecurity\s+audit\b",
    ]

    def __init__(
        self,
        local_model: str = DEFAULT_LOCAL_MODEL,
        local_base_url: str = DEFAULT_LOCAL_BASE_URL,
        cloud_model: str = DEFAULT_CLOUD_MODEL,
        enable_lightweight: bool = True,
        auto_fallback: bool = True,
    ):
        self.local_model = local_model
        self.local_base_url = local_base_url.rstrip("/")
        self.cloud_model = cloud_model
        self.enable_lightweight = enable_lightweight
        self.auto_fallback = auto_fallback
        self._planning_pattern = re.compile("|".join(self.PLANNING_KEYWORDS), re.IGNORECASE)

    def check_endpoint_health(self, timeout: float = 2.5) -> Tuple[bool, str]:
        """Probes the local Ollama/OpenAI-compatible server for connectivity and status.

        Returns:
            Tuple of (is_healthy, message).
        """
        # Try /v1/models (standard OpenAI endpoint)
        v1_url = f"{self.local_base_url}/models"
        # Try /api/tags (Ollama native endpoint if base_url has /v1)
        root_url = self.local_base_url.replace("/v1", "")
        native_url = f"{root_url}/api/tags"

        for url in [v1_url, native_url]:
            try:
                resp = requests.get(url, timeout=timeout)
                if resp.status_code == 200:
                    data = resp.json()
                    models = []
                    if "data" in data and isinstance(data["data"], list):
                        models = [m.get("id") for m in data["data"] if isinstance(m, dict)]
                    elif "models" in data and isinstance(data["models"], list):
                        models = [m.get("name") for m in data["models"] if isinstance(m, dict)]
                    
                    model_str = f"Available models: {models}" if models else "Endpoint reachable."
                    return True, f"Clean response from {url}. {model_str}"
            except requests.RequestException:
                continue

        return False, f"Could not connect to local server at {self.local_base_url} within {timeout}s."

    def classify_task(self, prompt: str) -> RouteDecision:
        """Analyzes prompt content to determine if local or cloud escalation is appropriate."""
        # 1. Check prompt length and complexity keywords
        if self._planning_pattern.search(prompt) or len(prompt.split()) > 250:
            return RouteDecision(
                target="cloud",
                complexity=TaskComplexity.CLOUD_PLANNING,
                reason="Task requires high-level architecture/strategic planning or deep reasoning.",
            )

        # 2. Default to local lightweight execution
        return RouteDecision(
            target="local",
            complexity=TaskComplexity.LOCAL_LIGHTWEIGHT,
            reason="Task matches local coding, refactoring, or tool execution profile.",
        )

    async def execute(
        self,
        prompt: str,
        force_mode: Optional[str] = None,  # "local", "cloud", or None for auto
    ) -> AsyncGenerator[str, None]:
        """Routes the prompt and streams tokens from the selected agent, falling back if needed."""
        # Determine initial routing
        if force_mode == "cloud":
            decision = RouteDecision("cloud", TaskComplexity.CLOUD_PLANNING, "Forced cloud mode.")
        elif force_mode == "local":
            decision = RouteDecision("local", TaskComplexity.LOCAL_LIGHTWEIGHT, "Forced local mode.")
        else:
            decision = self.classify_task(prompt)

        logger.info("Routing decision: %s (%s)", decision.target.upper(), decision.reason)

        if decision.target == "local":
            # Check local health before attempting invocation
            healthy, health_msg = self.check_endpoint_health()
            if not healthy:
                logger.warning("Local endpoint unhealthy: %s", health_msg)
                if self.auto_fallback:
                    yield f"[Router Notice: Local endpoint unreachable ({health_msg}). Escalating to Cloud ({self.cloud_model})...]\n\n"
                    async for token in self._run_cloud(prompt):
                        yield token
                    return
                else:
                    raise ConnectionError(f"Local LLM unavailable: {health_msg}")

            try:
                # Attempt local execution
                async for token in self._run_local(prompt):
                    yield token
                return
            except Exception as err:
                logger.error("Local inference failed: %s", err, exc_info=True)
                if self.auto_fallback:
                    yield f"\n\n[Router Fallback: Local execution encountered an error ({err}). Escalating to Cloud ({self.cloud_model})...]\n\n"
                    async for token in self._run_cloud(prompt):
                        yield token
                else:
                    raise

        else:
            # Route directly to cloud
            async for token in self._run_cloud(prompt):
                yield token

    async def _run_local(self, prompt: str) -> AsyncGenerator[str, None]:
        """Spawns an agent using LocalOpenAIAgentConfig with .lightweight()."""
        config = get_local_agent_config(
            model=self.local_model,
            base_url=self.local_base_url,
            lightweight=self.enable_lightweight,
        )
        async with Agent(config) as agent:
            response = await agent.chat(prompt)
            async for token in response:
                yield token

    async def _run_cloud(self, prompt: str) -> AsyncGenerator[str, None]:
        """Spawns a cloud agent for complex planning or escalation."""
        config = get_cloud_agent_config(
            model=self.cloud_model,
        )
        async with Agent(config) as agent:
            response = await agent.chat(prompt)
            async for token in response:
                yield token
