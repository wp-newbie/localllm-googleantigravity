"""Streamlit Web GUI for Antigravity Local LLM & Hybrid Agent.

Provides an interactive chat interface supporting:
- Local LLM inference (Ollama via LocalOpenAIAgentConfig with .lightweight())
- Workspace folder inclusion (e.g. F:\\Personal) for agent tools
- Hybrid task routing (local coding vs cloud strategic planning)
- Real-time streaming responses
- Live endpoint health monitoring and model discovery
"""

from __future__ import annotations

import asyncio
import os
import sys
from typing import AsyncGenerator, Iterator

import streamlit as st

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.agent_config import (
    DEFAULT_CLOUD_MODEL,
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_PROJECT_WORKSPACE,
    resolve_workspaces,
)
from router.hybrid_router import HybridAgentRouter, TaskComplexity

st.set_page_config(
    page_title="Antigravity Local LLM & Hybrid Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)


def sync_stream(async_gen: AsyncGenerator[str, None]) -> Iterator[str]:
    """Wraps an async generator for Streamlit's synchronous st.write_stream."""
    loop = asyncio.new_event_loop()
    try:
        while True:
            try:
                yield loop.run_until_complete(async_gen.__anext__())
            except StopAsyncIteration:
                break
    finally:
        loop.close()


# --- Sidebar: Configuration & Health ---
with st.sidebar:
    st.title("⚙️ Agent Settings")
    st.markdown("Configure local inference, project workspace, and cloud escalation.")

    st.subheader("📂 Project Workspace Folder")
    workspace_input = st.text_input(
        "Local Project Directory",
        value=DEFAULT_PROJECT_WORKSPACE or "F:\\Personal",
        help="Local codebase directory for the agent to inspect, read, and write (e.g. F:\\Personal).",
    )
    resolved_workspaces = resolve_workspaces(workspace_input)
    if resolved_workspaces:
        st.success(f"✅ Included Workspace: `{resolved_workspaces[0]}`")
    elif workspace_input:
        st.warning(f"⚠️ Directory not found: `{workspace_input}`")

    st.divider()

    st.subheader("Local Inference (Ollama / vLLM)")
    base_url = st.text_input(
        "OpenAI Base URL",
        value=DEFAULT_LOCAL_BASE_URL,
        help="OpenAI-compatible endpoint for Ollama or LM Studio (e.g. http://localhost:11434/v1).",
    )

    # Health check button
    router_probe = HybridAgentRouter(local_base_url=base_url)
    if st.button("🔍 Check Endpoint Health", use_container_width=True):
        is_healthy, msg = router_probe.check_endpoint_health()
        if is_healthy:
            st.success(f"Connected! {msg}")
        else:
            st.error(f"Offline: {msg}")

    # Discover models
    is_healthy, health_msg = router_probe.check_endpoint_health(timeout=1.5)
    default_models = ["qwen2.5-coder:7b", "gemma4:latest", "qwen2.5-coder:0.5b"]

    local_model = st.selectbox(
        "Local Model Target",
        options=default_models,
        index=0,
        help="Model identifier registered in your local server.",
    )

    lightweight = st.toggle(
        "Enable .lightweight() Optimization",
        value=True,
        help="Restricts tool bloat, disables subagents, and tunes compaction for local inference.",
    )

    st.divider()

    st.subheader("Routing & Cloud Escalation")
    mode_selection = st.radio(
        "Routing Mode",
        options=["⚡ Hybrid (Auto-Route & Fallback)", "🟢 Local Only", "☁️ Cloud Only"],
        index=0,
        help="Hybrid auto-routes tasks based on complexity and fails over to cloud on local errors.",
    )

    cloud_model = st.selectbox(
        "Cloud Frontier Model",
        options=["gemini-2.5-flash", "gemini-2.5-pro"],
        index=0,
        help="Frontier model used for high-level architecture planning and automatic fallback.",
    )

    auto_fallback = st.checkbox(
        "Automatic Cloud Fallback",
        value=True,
        help="Escalate to cloud model if local inference encounters connection errors or timeouts.",
    )

    st.divider()
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# --- Main Area ---
st.title("🤖 Antigravity Local LLM & Hybrid Agent")
st.caption(
    "Powered by **Google Antigravity SDK (`google-antigravity`)** | Connected to **Local Ollama** (`qwen2.5-coder:7b`)"
)

# Status bar metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Local Endpoint", "Online" if is_healthy else "Offline", delta=None)
with col2:
    st.metric("Local Model", local_model)
with col3:
    mode_label = (
        "HYBRID"
        if "Hybrid" in mode_selection
        else ("LOCAL" if "Local" in mode_selection else "CLOUD")
    )
    st.metric("Routing Mode", mode_label)
with col4:
    ws_display = os.path.basename(resolved_workspaces[0]) if resolved_workspaces else "None"
    st.metric("Workspace", ws_display)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "👋 Hello! I am your Antigravity Agent. I run locally on your **Ollama** server "
                "with `.lightweight()` optimizations for lightning-fast coding, and can seamlessly "
                "escalate complex architecture tasks to cloud models. How can I help you today?"
            ),
            "metadata": {"target": "local", "reason": "System welcome"},
        }
    ]

# Display existing messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "metadata" in msg and msg["metadata"]:
            target = msg["metadata"].get("target", "").upper()
            reason = msg["metadata"].get("reason", "")
            if target:
                color = "green" if target == "LOCAL" else "blue"
                st.caption(f":{color}[Routed to {target}]: {reason}")

# Quick action prompts
st.markdown("**Quick Prompts:**")
qp_cols = st.columns(3)
quick_prompt = None
with qp_cols[0]:
    if st.button("⚡ Local: Inspect Workspace Structure", use_container_width=True):
        quick_prompt = "Examine the files in the attached project workspace and give an overview of its architecture."
with qp_cols[1]:
    if st.button("⚡ Local: Fibonacci with Memoization", use_container_width=True):
        quick_prompt = "Write a Python function to calculate Fibonacci numbers using memoization."
with qp_cols[2]:
    if st.button("☁️ Cloud: System Architecture Plan", use_container_width=True):
        quick_prompt = "Design a comprehensive multi-phase architecture and roadmap for a distributed transaction engine."

user_input = st.chat_input("Enter your instruction, coding task, or architecture request...")
prompt = user_input or quick_prompt

if prompt:
    # Append and display user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Determine forced mode if not hybrid
    force_mode = None
    if "Local" in mode_selection:
        force_mode = "local"
    elif "Cloud" in mode_selection:
        force_mode = "cloud"

    # Instantiate router with resolved workspaces
    router = HybridAgentRouter(
        local_model=local_model,
        local_base_url=base_url,
        cloud_model=cloud_model,
        workspaces=resolved_workspaces,
        enable_lightweight=lightweight,
        auto_fallback=auto_fallback,
    )

    # Classify decision for user visibility
    if force_mode:
        target_name = force_mode.upper()
        reason = f"Forced {force_mode} mode by user selection."
    else:
        decision = router.classify_task(prompt)
        target_name = decision.target.upper()
        reason = decision.reason

    # Stream assistant response
    with st.chat_message("assistant"):
        badge_color = "green" if target_name == "LOCAL" else "blue"
        st.caption(f":{badge_color}[Routing Decision: {target_name}] — {reason}")

        stream_gen = router.execute(prompt, force_mode=force_mode)
        full_response = st.write_stream(sync_stream(stream_gen))

    # Save to history
    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": full_response,
            "metadata": {"target": target_name, "reason": reason},
        }
    )
