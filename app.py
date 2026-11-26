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
import json
import os
import sys
import time
from typing import AsyncGenerator, Iterator

import streamlit as st

# Ensure project root is on sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import importlib
import config.agent_config
import router.hybrid_router

importlib.reload(config.agent_config)
importlib.reload(router.hybrid_router)

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

def format_duration(seconds: float) -> str:
    """Formats elapsed execution seconds into Antigravity IDE style (e.g. '1m', '1m 12s', '4.2s')."""
    if seconds >= 60:
        m, s = divmod(int(seconds), 60)
        return f"{m}m" if s == 0 else f"{m}m {s}s"
    return f"{seconds:.1f}s"


def get_file_icon(filename: str) -> str:
    """Returns a visual icon for a given file extension."""
    ext = os.path.splitext(filename)[1].lower()
    icon_map = {
        ".py": "🐍",
        ".ts": "⚛️",
        ".tsx": "⚛️",
        ".js": "⚡",
        ".jsx": "⚡",
        ".cjs": "⚡",
        ".mjs": "⚡",
        ".json": "⚙️",
        ".md": "📖",
        ".html": "🌐",
        ".css": "🎨",
        ".env": "🔐",
        ".yaml": "🔧",
        ".yml": "🔧",
        ".sql": "🗄️",
    }
    return icon_map.get(ext, "📄")


def get_language(filename: str) -> str:
    """Maps file extension to Streamlit / Prism syntax highlighting language."""
    ext = os.path.splitext(filename)[1].lower()
    lang_map = {
        ".py": "python",
        ".ts": "typescript",
        ".tsx": "typescript",
        ".js": "javascript",
        ".jsx": "javascript",
        ".cjs": "javascript",
        ".mjs": "javascript",
        ".json": "json",
        ".md": "markdown",
        ".html": "html",
        ".css": "css",
        ".sql": "sql",
        ".sh": "bash",
        ".ps1": "powershell",
        ".yaml": "yaml",
        ".yml": "yaml",
    }
    return lang_map.get(ext, "text")


def scan_workspace_tree(root_dir: str, exclusions: list[str], max_depth: int = 4) -> list[str]:
    """Scans and lists files in workspace honoring folder exclusions."""
    tree = []
    root_dir = os.path.abspath(root_dir)
    excl_set = set(exclusions) if exclusions else set()
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in excl_set and not d.startswith(".")]
        depth = root[len(root_dir):].count(os.sep)
        if depth >= max_depth:
            dirs.clear()
            continue
        rel = os.path.relpath(root, root_dir)
        for f in files:
            if not f.startswith(".") or f in [".env", ".env.example", ".gitignore"]:
                tree.append(os.path.normpath(os.path.join(rel, f) if rel != "." else f))
    return sorted(tree)


def search_workspace_content(root_dir: str, query: str, exclusions: list[str], max_results: int = 20) -> list[tuple[str, str]]:
    """Searches file paths and contents in the workspace for query (Find in Files)."""
    results = []
    root_dir = os.path.abspath(root_dir)
    excl_set = set(exclusions) if exclusions else set()
    q_lower = query.lower()
    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if d not in excl_set and not d.startswith(".")]
        for f in files:
            ext = os.path.splitext(f)[1].lower()
            if ext in [".png", ".jpg", ".jpeg", ".vrm", ".xwear", ".exe", ".ico", ".woff", ".woff2", ".ttf"]:
                continue
            fpath = os.path.join(root, f)
            relpath = os.path.normpath(os.path.relpath(fpath, root_dir))
            if q_lower in relpath.lower():
                results.append((relpath, "Filename match"))
                if len(results) >= max_results:
                    return results
                continue
            try:
                with open(fpath, "r", encoding="utf-8", errors="ignore") as fp:
                    for i, line in enumerate(fp, 1):
                        if q_lower in line.lower():
                            results.append((relpath, f"L{i}: {line.strip()[:90]}"))
                            if len(results) >= max_results:
                                return results
                            break
            except Exception:
                pass
    return results


tab_chat, tab_explorer = st.tabs(["💬 Chat & Assistant", "📁 Workspace Code Explorer"])

with tab_chat:
    # Display existing messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg["role"] == "assistant" and msg.get("metadata", {}).get("worked_for"):
                with st.expander(f"Worked for {msg['metadata']['worked_for']}", expanded=False):
                    st.caption(f"🎯 **Model**: `{msg['metadata'].get('model', 'local')}` | 📋 **Mode**: `{msg['metadata'].get('target', 'LOCAL')}`")
                    st.caption(f"ℹ️ **Routing Reason**: {msg['metadata'].get('reason', '')}")
                    if msg["metadata"].get("workspace"):
                        st.caption(f"📂 **Workspace**: `{msg['metadata']['workspace']}`")
            elif msg["role"] == "assistant" and "metadata" in msg and msg["metadata"]:
                target = msg["metadata"].get("target", "").upper()
                reason = msg["metadata"].get("reason", "")
                if target and target != "SYSTEM":
                    color = "green" if target == "LOCAL" else "blue"
                    st.caption(f":{color}[Routed to {target}]: {reason}")

            st.markdown(msg["content"])

    # AI Workspace-Scanned Dynamic Quick Prompts
    suggested_prompts = scan_workspace_prompts(resolved_workspaces if include_workspace else None)
    selected_quick_prompt = None

    if suggested_prompts:
        ws_display = (
            ", ".join(os.path.basename(w.rstrip("/\\")) for w in resolved_workspaces)
            if (include_workspace and resolved_workspaces)
            else "Standalone Mode"
        )
        st.markdown(f"💡 **AI Scanned Quick Prompts** *(for `{ws_display}`)*:")
        cols = st.columns(len(suggested_prompts))
        for i, sp in enumerate(suggested_prompts):
            with cols[i]:
                if st.button(sp["label"], key=f"dynamic_qp_{i}", use_container_width=True):
                    selected_quick_prompt = sp["prompt"]

    user_input = st.chat_input("Enter your instruction, coding task, or architecture request...")
    pending_prompt = st.session_state.pop("pending_prompt", None)
    prompt = user_input or selected_quick_prompt or pending_prompt

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
            start_time = time.perf_counter()
            active_model = local_model if target_name == "LOCAL" else cloud_model

            with st.status(f"Thinking with {active_model}...", expanded=True) as status_box:
                status_box.write(f"🎯 **Routed To**: `{target_name}` ({active_model})")
                status_box.write(f"📋 **Task Analysis**: {reason}")
                if include_workspace and resolved_workspaces:
                    status_box.write(f"📂 **Attached Workspace**: `{', '.join(resolved_workspaces)}`")
                if full_ide_permissions:
                    status_box.write("🛡️ **IDE Parity**: Autonomous Permissions Active (`policy.allow_all`)")

                try:
                    stream_gen = router.execute(prompt, force_mode=force_mode)
                    full_response = st.write_stream(sync_stream(stream_gen))
                    duration_str = format_duration(time.perf_counter() - start_time)
                    status_box.update(label=f"Worked for {duration_str}", state="complete", expanded=False)

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": full_response,
                            "metadata": {
                                "target": target_name,
                                "reason": reason,
                                "worked_for": duration_str,
                                "model": active_model,
                                "workspace": ", ".join(resolved_workspaces) if (include_workspace and resolved_workspaces) else None,
                            },
                        }
                    )
                except ConnectionError as ce:
                    status_box.update(label="Connection Error", state="error", expanded=True)
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
                    status_box.update(label="Execution Error", state="error", expanded=True)
                    st.error(f"Execution Error: {e}")

with tab_explorer:
    if not (include_workspace and resolved_workspaces):
        st.info("ℹ️ Workspace is currently excluded or not configured. Enable **Include Local Workspace** in the sidebar to browse project code.")
    else:
        active_ws = resolved_workspaces[0] if len(resolved_workspaces) == 1 else st.selectbox(
            "Select Workspace to Explore",
            options=resolved_workspaces,
            format_func=lambda x: f"📁 {os.path.basename(x)} ({x})",
        )

        st.caption(f"Active Workspace Root: `{active_ws}`")

        col_search, col_stats = st.columns([3, 1])
        with col_search:
            search_query = st.text_input("🔍 Find in Files / Search Code", placeholder="Search filename or code text (e.g. electron, fetchLmStudio, def, import)...")
        with col_stats:
            all_files = scan_workspace_tree(active_ws, selected_exclusions)
            st.metric("Total Project Files", len(all_files))

        col_tree, col_view = st.columns([1, 2])
        selected_rel_path = None

        with col_tree:
            st.subheader("📑 Project Files")
            if search_query.strip():
                matches = search_workspace_content(active_ws, search_query.strip(), selected_exclusions)
                if matches:
                    st.caption(f"Found {len(matches)} match(es):")
                    match_display = [f"{get_file_icon(m[0])} {m[0]} ({m[1]})" for m in matches]
                    match_idx = st.selectbox("Search Results", range(len(match_display)), format_func=lambda i: match_display[i], key="search_sel")
                    selected_rel_path = matches[match_idx][0]
                else:
                    st.warning("No matches found in project.")
            else:
                if all_files:
                    file_options = [f"{get_file_icon(f)} {f}" for f in all_files]
                    sel_idx = st.selectbox("Browse Files", range(len(file_options)), format_func=lambda i: file_options[i], key="tree_sel")
                    selected_rel_path = all_files[sel_idx]
                else:
                    st.info("No files available in this workspace.")

        with col_view:
            if selected_rel_path:
                full_path = os.path.join(active_ws, selected_rel_path)
                lang = get_language(selected_rel_path)
                icon = get_file_icon(selected_rel_path)

                st.subheader(f"{icon} `{selected_rel_path}`")

                try:
                    size_kb = os.path.getsize(full_path) / 1024
                    with open(full_path, "r", encoding="utf-8", errors="replace") as f:
                        content = f.read()
                    lines = len(content.splitlines())

                    st.caption(f"📏 **{lines}** lines | 💾 **{size_kb:.1f} KB** | 🔤 Syntax: `{lang}`")

                    # AI Action Buttons
                    st.markdown("**🤖 AI Actions on this File (Local LLM):**")
                    act_col1, act_col2, act_col3 = st.columns(3)
                    with act_col1:
                        if st.button("🧠 Explain File", key="ai_act_explain", use_container_width=True):
                            st.session_state.pending_prompt = (
                                f"Please analyze and explain the architecture, functions, and key logic of `{selected_rel_path}` in `{os.path.basename(active_ws)}`:\n\n"
                                f"```{lang}\n{content[:4000]}\n```"
                            )
                            st.rerun()
                    with act_col2:
                        if st.button("🔍 Code Review", key="ai_act_review", use_container_width=True):
                            st.session_state.pending_prompt = (
                                f"Please conduct a comprehensive code review of `{selected_rel_path}`. Identify potential bugs, edge cases, performance improvements, and security considerations:\n\n"
                                f"```{lang}\n{content[:4000]}\n```"
                            )
                            st.rerun()
                    with act_col3:
                        if st.button("🧪 Generate Tests", key="ai_act_test", use_container_width=True):
                            st.session_state.pending_prompt = (
                                f"Write a complete unit test suite covering key edge cases for the exports and functions in `{selected_rel_path}`:\n\n"
                                f"```{lang}\n{content[:4000]}\n```"
                            )
                            st.rerun()

                    # Code view
                    st.code(content, language=lang, line_numbers=True)
                except Exception as read_err:
                    st.error(f"Could not read file `{selected_rel_path}`: {read_err}")
            else:
                st.info("Select a file from the project tree to view its contents and trigger AI actions.")
