"""Streamlit Web GUI for Antigravity Local LLM & Hybrid Agent.

Provides an interactive chat interface supporting:
- Local LLM inference via LM Studio (using LocalOpenAIAgentConfig with .lightweight())
- Dynamic model selection based on available LM Studio models (Qwen 2.5, Gemma 4, etc.)
- Multi-workspace selection from ANTIGRAVITY_WORKSPACE environment / config
- Include/Exclude workspace folder toggle and subfolder exclusion controls
- Hybrid task routing (local coding vs cloud strategic planning)
- Real-time streaming responses with graceful connection recovery
- Live endpoint health monitoring
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
from typing import AsyncGenerator, Iterator

import streamlit as st

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from config.agent_config import (
        DEFAULT_CLOUD_MODEL,
        DEFAULT_EXCLUDED_FOLDERS,
        DEFAULT_LM_STUDIO_BASE_URL,
        DEFAULT_LOCAL_BASE_URL,
        DEFAULT_LOCAL_MODEL,
        DEFAULT_PROJECT_WORKSPACE,
        get_available_local_models,
        get_configured_workspaces,
        get_lm_studio_models,
        resolve_workspaces,
    )
except ImportError:
    import importlib
    import config.agent_config
    importlib.reload(config.agent_config)
    from config.agent_config import (
        DEFAULT_CLOUD_MODEL,
        DEFAULT_EXCLUDED_FOLDERS,
        DEFAULT_LM_STUDIO_BASE_URL,
        DEFAULT_LOCAL_BASE_URL,
        DEFAULT_LOCAL_MODEL,
        DEFAULT_PROJECT_WORKSPACE,
        get_available_local_models,
        get_configured_workspaces,
        get_lm_studio_models,
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
    st.markdown("Configure local models (LM Studio), workspaces, and cloud routing.")

    # 1. Project Workspace Section
    st.subheader("📂 Project Workspace Folder")
    include_workspace = st.toggle(
        "Include Local Workspace",
        value=True,
        help="Toggle ON to attach project folders for the agent to inspect/edit. Toggle OFF to run in standalone mode.",
    )

    resolved_workspaces: list[str] | None = None
    selected_exclusions = DEFAULT_EXCLUDED_FOLDERS

    if include_workspace:
        configured_workspaces = get_configured_workspaces()
        if not configured_workspaces:
            configured_workspaces = ["F:\\Personal", "F:\\Code"]

        attach_multiple = st.checkbox(
            "Attach Multiple Workspaces",
            value=False,
            help="Enable to select more than one workspace folder simultaneously.",
        )

        if attach_multiple:
            selected_list = st.multiselect(
                "Select Workspaces to Include",
                options=configured_workspaces,
                default=[configured_workspaces[0]] if configured_workspaces else [],
                help="Choose from workspaces configured via ANTIGRAVITY_WORKSPACE.",
            )
            custom_extra = st.text_input(
                "Additional Custom Path (Optional)",
                value="",
                placeholder="e.g. F:\\AnotherProject",
            )
            candidates = list(selected_list)
            if custom_extra.strip():
                candidates.append(custom_extra.strip())
            resolved_workspaces = resolve_workspaces(candidates, enabled=True)
        else:
            options = configured_workspaces + ["Custom Path..."]
            chosen = st.selectbox(
                "Select Workspace (from ANTIGRAVITY_WORKSPACE)",
                options=options,
                index=0,
                help="Select from directories defined in your ANTIGRAVITY_WORKSPACE settings.",
            )

            if chosen == "Custom Path...":
                custom_input = st.text_input(
                    "Enter Custom Directory Path",
                    value="",
                    placeholder="e.g. F:\\Projects\\MyCode",
                )
                resolved_workspaces = resolve_workspaces(custom_input, enabled=True)
            else:
                resolved_workspaces = resolve_workspaces(chosen, enabled=True)

        if resolved_workspaces:
            for ws in resolved_workspaces:
                st.success(f"✅ Included: `{ws}`")
        else:
            st.info("ℹ️ No active workspace selected or directory not found.")

        all_exclude_options = [
            "node_modules",
            "dist",
            "release",
            ".git",
            ".venv",
            "build",
            "__pycache__",
            "coverage",
            "temp",
        ]
        selected_exclusions = st.multiselect(
            "Exclude Subfolders (Ignored)",
            options=all_exclude_options,
            default=[x for x in DEFAULT_EXCLUDED_FOLDERS if x in all_exclude_options],
            help="Subdirectories the agent will strictly ignore to avoid wasting context tokens.",
        )
    else:
        st.info("🚫 **Workspace Excluded**: Agent runs in standalone sandbox mode without access to local project files.")

    st.divider()

    # 2. Local Inference via LM Studio
    st.subheader("🖥️ Local LLM (LM Studio)")
    base_url = st.text_input(
        "LM Studio Base URL",
        value=DEFAULT_LM_STUDIO_BASE_URL,
        help="OpenAI-compatible endpoint for LM Studio (default: http://localhost:1234/v1).",
    )

    # Health check
    router_probe = HybridAgentRouter(local_base_url=base_url)
    is_healthy, health_msg = router_probe.check_endpoint_health(timeout=1.5)

    if is_healthy:
        st.success("🟢 LM Studio server is online and ready!")
    else:
        st.error(f"⚠️ LM Studio server at `{base_url}` is offline.")
        st.caption("Tip: In LM Studio, click the **Local Server (↔)** tab on the left and switch the server to **ON**.")

    if st.button("🔍 Check Server Health", use_container_width=True):
        is_h, msg = router_probe.check_endpoint_health()
        if is_h:
            st.success(f"Connected! {msg}")
        else:
            st.error(f"Offline: {msg}")

    # Discover LM Studio models
    available_models = get_lm_studio_models(base_url)
    if not available_models:
        available_models = ["qwen2.5-7b-instruct-1m", "gemma-4-e4b-it"]

    model_options = available_models + ["Custom Model Name..."]
    chosen_model = st.selectbox(
        "Available LM Studio Models",
        options=model_options,
        index=0,
        help="Discovered models from LM Studio memory and local library.",
    )

    if chosen_model == "Custom Model Name...":
        local_model = st.text_input("Enter Model Tag/Identifier", value="qwen2.5-7b-instruct-1m")
    else:
        local_model = chosen_model

    full_ide_permissions = st.toggle(
        "Full Antigravity IDE Permissions",
        value=True,
        help="Grants local LLM full autonomous tool execution permissions matching Antigravity IDE (file reading, editing, creating, directory traversal, code search, and terminal commands).",
    )

    if full_ide_permissions:
        st.caption("🛡️ **IDE Parity Active**: `policy.allow_all()` + Autonomous Terminal & File Tools")

    lightweight = st.toggle(
        "Enable .lightweight() Optimization",
        value=False,
        help="Tunes compaction and focuses on 4 core development tools for smaller local models.",
    )

    st.divider()

    # 3. Routing Mode & Cloud Escalation
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
    f"Powered by **Google Antigravity SDK (`google-antigravity`)** | Connected to **LM Studio** (`{local_model}`)"
)

# Status bar metrics
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("LM Studio", "Online" if is_healthy else "Offline")
with col2:
    st.metric("Active Model", local_model)
with col3:
    mode_label = (
        "HYBRID"
        if "Hybrid" in mode_selection
        else ("LOCAL" if "Local" in mode_selection else "CLOUD")
    )
    st.metric("Routing Mode", mode_label)
with col4:
    if include_workspace and resolved_workspaces:
        if len(resolved_workspaces) == 1:
            ws_display = os.path.basename(resolved_workspaces[0])
        else:
            ws_display = f"{len(resolved_workspaces)} Workspaces"
    elif include_workspace:
        ws_display = "None (Empty)"
    else:
        ws_display = "Excluded"
    st.metric("Workspace", ws_display)

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                f"👋 Hello! I am your Antigravity Agent running on **LM Studio** "
                f"using **`{local_model}`** with `.lightweight()` optimizations for lightning-fast coding, "
                "and can seamlessly escalate complex architecture tasks to cloud models. How can I help you today?"
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
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    force_mode = None
    if "Local" in mode_selection:
        force_mode = "local"
    elif "Cloud" in mode_selection:
        force_mode = "cloud"

    router = HybridAgentRouter(
        local_model=local_model,
        local_base_url=base_url,
        cloud_model=cloud_model,
        workspaces=resolved_workspaces,
        include_workspace=include_workspace,
        exclude_folders=selected_exclusions,
        enable_lightweight=lightweight,
        full_ide_permissions=full_ide_permissions,
        auto_fallback=auto_fallback,
    )

    if force_mode:
        target_name = force_mode.upper()
        reason = f"Forced {force_mode} mode by user selection."
    else:
        decision = router.classify_task(prompt)
        target_name = decision.target.upper()
        reason = decision.reason

    with st.chat_message("assistant"):
        badge_color = "green" if target_name == "LOCAL" else "blue"
        st.caption(f":{badge_color}[Routing Decision: {target_name}] — {reason}")

        try:
            stream_gen = router.execute(prompt, force_mode=force_mode)
            full_response = st.write_stream(sync_stream(stream_gen))

            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": full_response,
                    "metadata": {"target": target_name, "reason": reason},
                }
            )
        except ConnectionError as ce:
            error_msg = (
                f"⚠️ **Could not connect to LM Studio server at `{base_url}`.**\n\n"
                f"- **Quick Fix**: Open LM Studio, click the **Local Server (↔)** tab, and ensure the server toggle is **ON** (port 1234).\n"
                f"- **Or**: Switch the routing mode in the sidebar to **⚡ Hybrid** or **☁️ Cloud Only**."
            )
            st.error(error_msg)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg,
                    "metadata": {"target": "ERROR", "reason": str(ce)},
                }
            )
        except Exception as e:
            st.error(f"Execution Error: {e}")
