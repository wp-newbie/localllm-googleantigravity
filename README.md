# Antigravity Local LLM & Hybrid Router Integration

This project integrates the **Google Antigravity SDK (`google-antigravity`)** with a local inference server (**Ollama** or any OpenAI-compatible runtime such as LM Studio or vLLM) using `LocalOpenAIAgentConfig` with `.lightweight()` optimizations, coupled with an intelligent **Hybrid & Fallback Router** that escalates complex planning tasks to cloud models.

It includes both a **CLI test runner** and a full **interactive Streamlit Web GUI (`app.py`)**.

---

## Architecture Overview

```
                          ┌────────────────────────┐
                          │   User Prompt / Task   │
                          │   (CLI or Streamlit)   │
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

1. **Interactive Web GUI (`app.py`)**:
   - Modern browser interface with chat history, model selection, live endpoint health monitoring, and real-time streaming tokens.
   - Live badge indicators showing routing decisions (`LOCAL` vs `CLOUD`).
   - One-click quick prompts for coding vs strategic planning.

2. **`LocalOpenAIAgentConfig` & `.lightweight()`** (`config/agent_config.py`):
   - Connects Antigravity's autonomous agent harness to `http://localhost:11434/v1`.
   - `.lightweight()` strips prompt overhead, restricts the tool set to core filesystem/command tools (`run_command`, `view_file`, `create_file`, `edit_file`), disables heavy subagent recursion, and tunes memory compaction thresholds (`token_threshold=65536`) to fit tightly constrained local KV caches.

3. **`HybridAgentRouter`** (`router/hybrid_router.py`):
   - **Task Classification**: Routes coding, syntax fixes, and file operations to the local model; escalates architecture, system design, and multi-phase roadmaps to frontier cloud models.
   - **Circuit Breaker & Automatic Fallback**: Probes endpoint health; if the local server is down, times out, or fails during inference, transparently falls back to the cloud model without dropping the user's task.

4. **CLI Runner** (`cli.py`):
   - A command-line utility for testing endpoint health, running single prompts, and switching between `local`, `cloud`, and `hybrid` modes.

---

## Directory Structure

```
antigravity-local-llm/
├── app.py                      # Interactive Streamlit Web GUI
├── cli.py                      # CLI test runner
├── config/
│   ├── __init__.py
│   └── agent_config.py         # LocalOpenAIAgentConfig & CloudAgentConfig factories
├── router/
│   ├── __init__.py
│   └── hybrid_router.py        # Intelligent routing, health check & fallback engine
├── pyproject.toml              # Project metadata & dependencies
├── requirements.txt            # Package requirements
└── README.md                   # Documentation & setup guide
```

---

## Quickstart

### 1. Prerequisites & Environment

Activate the virtual environment:

```powershell
cd C:\Users\<username>\.gemini\antigravity\scratch\antigravity-local-llm
# Or using the user profile variable:
# cd $env:USERPROFILE\.gemini\antigravity\scratch\antigravity-local-llm

.venv\Scripts\activate
```

### 2. Launch the Web GUI (Recommended)

Run the Streamlit application:

```powershell
streamlit run app.py
```

Your browser will automatically open to `http://localhost:8501`, giving you an interactive chat interface with live model status and routing badges!

### 3. Verify Local Ollama Endpoint (CLI)

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

### 4. Run Agent Invocations via CLI

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

---

## Programmatic Usage

You can import the modular configurations directly into your own scripts or pipelines:

```python
import asyncio
from google.antigravity import Agent
from config.agent_config import get_local_agent_config

async def main():
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
