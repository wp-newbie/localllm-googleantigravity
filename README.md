# Antigravity Local LLM & Hybrid Router Integration

This project integrates the **Google Antigravity SDK (`google-antigravity`)** with a local inference server (**Ollama** or any OpenAI-compatible runtime such as LM Studio or vLLM) using `LocalOpenAIAgentConfig` with `.lightweight()` optimizations, coupled with an intelligent **Hybrid & Fallback Router** that escalates complex planning tasks to cloud models.

---

## Architecture Overview

```
                          ┌────────────────────────┐
                          │   User Prompt / Task   │
                          └───────────┬────────────┘
                                      │
                                      ▼
                        ┌───────────────────────────┐
                        │    HybridAgentRouter      │
                        │  (Complexity Classifier)  │
                        └──────┬─────────────┬──────┘
                               │             │
        Simple Code / Edits    │             │   Architecture / Planning
        (or Local Mode)        ▼             ▼   (or Cloud Escalation)
                  ┌──────────────────────┐  ┌──────────────────────┐
                  │LocalOpenAIAgentConfig│  │   LocalAgentConfig   │
                  │   .lightweight()     │  │   (Cloud Gemini)     │
                  └──────────┬───────────┘  └──────────┬───────────┘
                             │                         │
                             ▼                         ▼
                  ┌──────────────────────┐  ┌──────────────────────┐
                  │     Local Ollama     │  │   Cloud Frontier     │
                  │http://localhost:11434│  │    (Gemini 2.5)      │
                  │  (qwen2.5-coder:7b)  │  │                      │
                  └──────────────────────┘  └──────────────────────┘
                             │ (Fallback on Failure/Unreachable)
                             └─────────────────────────►
```

### Key Components

1. **`LocalOpenAIAgentConfig` & `.lightweight()`** (`config/agent_config.py`):
   - Connects Antigravity's autonomous agent harness to `http://localhost:11434/v1`.
   - `.lightweight()` strips prompt overhead, restricts the tool set to core filesystem/command tools (`run_command`, `view_file`, `create_file`, `edit_file`), disables heavy subagent recursion, and tunes memory compaction thresholds (`token_threshold=65536`) to fit tightly constrained local KV caches.

2. **`HybridAgentRouter`** (`router/hybrid_router.py`):
   - **Task Classification**: Routes coding, syntax fixes, and file operations to the local model; escalates architecture, system design, and multi-phase roadmaps to frontier cloud models.
   - **Circuit Breaker & Automatic Fallback**: Probes endpoint health; if the local server is down, times out, or fails during inference, transparently falls back to the cloud model without dropping the user's task.

3. **CLI Runner** (`cli.py`):
   - A command-line utility for testing endpoint health, running single prompts, and switching between `local`, `cloud`, and `hybrid` modes.

---

## Directory Structure

```
antigravity-local-llm/
├── config/
│   ├── __init__.py
│   └── agent_config.py         # LocalOpenAIAgentConfig & CloudAgentConfig factories
├── router/
│   ├── __init__.py
│   └── hybrid_router.py        # Intelligent routing, health check & fallback engine
├── cli.py                      # CLI test runner
├── pyproject.toml              # Project metadata & dependencies
├── requirements.txt            # Package requirements
└── README.md                   # Documentation & setup guide
```

---

## Quickstart

### 1. Prerequisites & Environment

Ensure Python 3.10+ is available. Activate the virtual environment:

```powershell
cd C:\Users\tobsk\.gemini\antigravity\scratch\antigravity-local-llm
.venv\Scripts\activate
```

Dependencies installed:
- `google-antigravity>=0.1.24`
- `requests>=2.31.0`

### 2. Verify Local Ollama Endpoint

Run the CLI endpoint verification flag:

```powershell
python cli.py --test-endpoint
```

Sample output:
```
[*] Probing local endpoint: http://localhost:11434/v1 ...
[+] SUCCESS: Local endpoint is alive and responsive! (2101.8ms)
    Details: Clean response from http://localhost:11434/v1/models. Available models: ['qwen2.5-coder:7b', 'gemma4:latest']
```

### 3. Run Agent Invocations

#### A. Local Inference Mode
```powershell
python cli.py --mode local --model "qwen2.5-coder:7b" --prompt "Write a Python function to check for prime numbers."
```

#### B. Hybrid Mode (Automatic Escalation)
```powershell
# Routes to local model for coding:
python cli.py --mode hybrid --prompt "Write a quick binary search implementation."

# Routes to cloud model for high-level architectural planning:
python cli.py --mode hybrid --prompt "Design a comprehensive multi-phase architecture and roadmap for a distributed transaction engine."
```

#### C. Custom Model Target
```powershell
python cli.py --model "gemma4" --prompt "Explain how memoization reduces Fibonacci time complexity."
```

---

## Programmatic Usage

You can import the modular configurations directly into your own scripts or pipelines:

```python
import asyncio
from google.antigravity import Agent
from config.agent_config import get_local_agent_config

async def main():
    # Modular local configuration with .lightweight() preset
    config = get_local_agent_config(
        model="qwen2.5-coder:7b",
        base_url="http://localhost:11434/v1",
        lightweight=True,
    )

    async with Agent(config) as agent:
        response = await agent.chat("Refactor the error handling in our utility script.")
        async for token in response:
            print(token, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```

Or use the hybrid router for automatic escalation and error resilience:

```python
import asyncio
from router.hybrid_router import HybridAgentRouter

async def main():
    router = HybridAgentRouter(
        local_model="qwen2.5-coder:7b",
        cloud_model="gemini-2.5-flash",
        auto_fallback=True,
    )

    prompt = "Create a database schema and architectural plan for real-time telemetry ingestion."
    async for token in router.execute(prompt):
        print(token, end="", flush=True)

if __name__ == "__main__":
    asyncio.run(main())
```
