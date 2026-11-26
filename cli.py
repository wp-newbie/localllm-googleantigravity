"""CLI test and demonstration script for Antigravity Local LLM integration.

Validates local endpoint health, executes queries through LocalOpenAIAgentConfig,
and demonstrates hybrid routing and fallback capabilities.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time

from config.agent_config import (
    DEFAULT_CLOUD_MODEL,
    DEFAULT_LOCAL_BASE_URL,
    DEFAULT_LOCAL_MODEL,
    get_cloud_agent_config,
    get_local_agent_config,
)
from router.hybrid_router import HybridAgentRouter


def print_banner():
    banner = r"""
==============================================================
   Antigravity SDK Local LLM & Hybrid Router Runner
   - Local: LocalOpenAIAgentConfig (.lightweight())
   - Endpoint: Ollama (OpenAI-compatible /v1 API)
   - Fallback: Cloud Escalation Architecture
==============================================================
"""
    print(banner)


def check_endpoint(base_url: str) -> bool:
    """Verifies that the target local Ollama / OpenAI-compatible endpoint responds cleanly."""
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
        print("    Tip: Ensure Ollama is running (`ollama serve`) and listening at this address.")
        return False


async def run_chat(
    prompt: str,
    mode: str,
    model: str,
    base_url: str,
    cloud_model: str,
    lightweight: bool,
    auto_fallback: bool,
):
    """Executes a chat request via the HybridAgentRouter."""
    router = HybridAgentRouter(
        local_model=model,
        local_base_url=base_url,
        cloud_model=cloud_model,
        enable_lightweight=lightweight,
        auto_fallback=auto_fallback,
    )

    print(f"\n--- Running Execution ---")
    print(f"Mode          : {mode.upper()}")
    print(f"Local Model   : {model} (Endpoint: {base_url})")
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
        help="Quickly verify if the local Ollama server at http://localhost:11434 responds cleanly.",
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
        default=DEFAULT_LOCAL_MODEL,
        help=f"Target local model name in Ollama (default: {DEFAULT_LOCAL_MODEL}).",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        default=DEFAULT_LOCAL_BASE_URL,
        help=f"Local server base URL (default: {DEFAULT_LOCAL_BASE_URL}).",
    )
    parser.add_argument(
        "--cloud-model",
        type=str,
        default=DEFAULT_CLOUD_MODEL,
        help=f"Cloud model target for planning and fallback (default: {DEFAULT_CLOUD_MODEL}).",
    )
    parser.add_argument(
        "--no-lightweight",
        action="store_true",
        help="Disable .lightweight() optimization on LocalOpenAIAgentConfig.",
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable automatic escalation to cloud model when local inference is unavailable.",
    )

    args = parser.parse_args()
    print_banner()

    if args.test_endpoint:
        success = check_endpoint(args.base_url)
        sys.exit(0 if success else 1)

    asyncio.run(
        run_chat(
            prompt=args.prompt,
            mode=args.mode,
            model=args.model,
            base_url=args.base_url,
            cloud_model=args.cloud_model,
            lightweight=not args.no_lightweight,
            auto_fallback=not args.no_fallback,
        )
    )


if __name__ == "__main__":
    main()
