"""CLI test and demonstration script for Antigravity Local LLM integration.

Validates local endpoint health, executes queries through LocalOpenAIAgentConfig,
and demonstrates hybrid routing and fallback capabilities with LM Studio & Ollama model support.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time

from config.agent_config import (
    DEFAULT_CLOUD_MODEL,
    DEFAULT_EXCLUDED_FOLDERS,
    DEFAULT_LM_STUDIO_BASE_URL,
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    DEFAULT_OLLAMA_BASE_URL,
    DEFAULT_PROJECT_WORKSPACE,
    get_available_local_models,
    get_cloud_agent_config,
    get_local_agent_config,
    resolve_workspaces,
)
from router.hybrid_router import HybridAgentRouter


def print_banner():
    banner = r"""
==============================================================
   Antigravity SDK Local LLM & Hybrid Router Runner
   - Local: LocalOpenAIAgentConfig (.lightweight())
   - Endpoints: LM Studio (:1234) & Ollama (:11434)
   - Fallback: Cloud Escalation Architecture
==============================================================
"""
    print(banner)


def check_endpoint(base_url: str) -> bool:
    """Verifies that the target local Ollama / LM Studio endpoint responds cleanly."""
    router = HybridAgentRouter(local_base_url=base_url)
    print(f"[*] Probing local endpoint: {base_url} ...")
    start = time.perf_counter()
    healthy, message = router.check_endpoint_health()
    elapsed = (time.perf_counter() - start) * 1000

    if healthy:
        print(f"[+] SUCCESS: Local endpoint is alive and responsive! ({elapsed:.1f}ms)")
        print(f"    Details: {message}")
        return True
    else:
        print(f"[-] WARNING: Endpoint check failed ({elapsed:.1f}ms).")
        print(f"    Details: {message}")
        print("    Tip: Ensure your server (LM Studio or Ollama) is running and listening at this address.")
        return False


async def run_chat(
    prompt: str,
    mode: str,
    model: str,
    base_url: str,
    cloud_model: str,
    workspaces: list[str] | None,
    include_workspace: bool,
    exclude_folders: list[str] | None,
    lightweight: bool = False,
    full_ide_permissions: bool = True,
    auto_fallback: bool = True,
):
    """Executes a chat request via the HybridAgentRouter."""
    router = HybridAgentRouter(
        local_model=model,
        local_base_url=base_url,
        cloud_model=cloud_model,
        workspaces=workspaces,
        include_workspace=include_workspace,
        exclude_folders=exclude_folders,
        enable_lightweight=lightweight,
        full_ide_permissions=full_ide_permissions,
        auto_fallback=auto_fallback,
    )

    ws_status = router.workspaces if (include_workspace and router.workspaces) else "None (Excluded)"
    print(f"\n--- Running Execution ---")
    print(f"Mode          : {mode.upper()}")
    print(f"Local Model   : {model} (Endpoint: {base_url})")
    print(f"IDE Parity    : {full_ide_permissions} (policy.allow_all + autonomous tools)")
    print(f"Workspaces    : {ws_status}")
    print(f"Excluded Dirs : {router.exclude_folders if include_workspace else 'N/A'}")
    print(f"Lightweight   : {lightweight}")
    print(f"Cloud Fallback: {cloud_model} (Enabled: {auto_fallback})")
    print(f"Prompt        : {prompt}")
    print("-------------------------\n")

    force_mode = None if mode == "hybrid" else mode
    print("Response:\n")
    try:
        async for token in router.execute(prompt, force_mode=force_mode):
            sys.stdout.write(token)
            sys.stdout.flush()
        print("\n\n[+] Execution completed successfully.")
    except Exception as e:
        print(f"\n[-] Execution failed with error: {e}", file=sys.stderr)
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Test script for Antigravity SDK local LLM integration and hybrid router."
    )
    parser.add_argument(
        "--test-endpoint",
        action="store_true",
        help="Quickly verify if the local server responds cleanly.",
    )
    parser.add_argument(
        "--provider",
        choices=["lmstudio", "ollama"],
        default="lmstudio",
        help="Local LLM provider: 'lmstudio' (default: :1234) or 'ollama' (default: :11434).",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List all models discovered from LM Studio or Ollama.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Write a clean Python function to calculate the Fibonacci series using memoization.",
        help="Prompt to send to the agent.",
    )
    parser.add_argument(
        "--mode",
        choices=["hybrid", "local", "cloud"],
        default="hybrid",
        help="Execution mode: 'hybrid' (smart routing & fallback), 'local', or 'cloud'.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Target local model name (defaults to active provider model e.g. qwen2.5-7b-instruct-1m).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=None,
        help="Custom local server base URL.",
    )
    parser.add_argument(
        "--cloud-model",
        type=str,
        default=DEFAULT_CLOUD_MODEL,
        help=f"Cloud model target for planning and fallback (default: {DEFAULT_CLOUD_MODEL}).",
    )
    parser.add_argument(
        "--workspace",
        "-w",
        type=str,
        default=DEFAULT_PROJECT_WORKSPACE,
        help=f"Local project folder to include in the agent's workspace (default: {DEFAULT_PROJECT_WORKSPACE}).",
    )
    parser.add_argument(
        "--no-workspace",
        action="store_true",
        help="Explicitly exclude and detach local project workspace.",
    )
    parser.add_argument(
        "--exclude-folders",
        type=str,
        default=None,
        help="Comma-separated list of folder names to strictly ignore (e.g. 'node_modules,dist,.git').",
    )
    parser.add_argument(
        "--no-lightweight",
        action="store_true",
        help="Disable .lightweight() optimization on LocalOpenAIAgentConfig.",
    )
    parser.add_argument(
        "--no-full-permissions",
        action="store_true",
        help="Disable full Antigravity IDE permissions parity.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable automatic escalation to cloud model when local inference is unavailable.",
    )

    args = parser.parse_args()
    print_banner()

    # Determine default base_url & model based on provider
    if args.provider == "lmstudio":
        base_url = args.base_url or DEFAULT_LM_STUDIO_BASE_URL
        discovered = get_available_local_models("lmstudio", base_url)
        model = args.model or (discovered[0] if discovered else "qwen2.5-7b-instruct-1m")
    else:
        base_url = args.base_url or DEFAULT_OLLAMA_BASE_URL
        discovered = get_available_local_models("ollama", base_url)
        model = args.model or (discovered[0] if discovered else DEFAULT_LOCAL_MODEL)

    if args.list_models:
        print(f"[*] Discovered models for provider '{args.provider}' ({base_url}):")
        for m in discovered:
            print(f"    - {m}")
        sys.exit(0)

    if args.test_endpoint:
        success = check_endpoint(base_url)
        sys.exit(0 if success else 1)

    include_workspace = not args.no_workspace
    workspaces = resolve_workspaces(args.workspace, enabled=include_workspace)

    exclude_list = None
    if args.exclude_folders:
        exclude_list = [x.strip() for x in args.exclude_folders.split(",") if x.strip()]

    asyncio.run(
        run_chat(
            prompt=args.prompt,
            mode=args.mode,
            model=model,
            base_url=base_url,
            cloud_model=args.cloud_model,
            workspaces=workspaces,
            include_workspace=include_workspace,
            exclude_folders=exclude_list,
            lightweight=not args.no_lightweight,
            full_ide_permissions=not args.no_full_permissions,
            auto_fallback=not args.no_fallback,
        )
    )


if __name__ == "__main__":
    main()
