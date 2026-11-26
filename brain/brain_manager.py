"""Brain and Core Memories management module for Antigravity Local LLM Agent.

Provides persistent memory storage, contextual memory injection into agent prompts,
and conversation state checkpointing for seamless backtracking.
"""

from __future__ import annotations

import datetime
import json
import logging
import os
import uuid
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_BRAIN_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".brain"
)


class BrainManager:
    """Manages persistent core memories and backtracking checkpoints for the local AI."""

    def __init__(self, brain_dir: Optional[str] = None):
        self.brain_dir = brain_dir or DEFAULT_BRAIN_DIR
        os.makedirs(self.brain_dir, exist_ok=True)
        self.memories_file = os.path.join(self.brain_dir, "core_memories.json")
        self.checkpoints_file = os.path.join(self.brain_dir, "checkpoints.json")
        self._ensure_storage()

    def _ensure_storage(self) -> None:
        """Initializes storage files with default seed memories if new."""
        if not os.path.exists(self.memories_file):
            default_memories = [
                {
                    "id": str(uuid.uuid4())[:8],
                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "category": "infrastructure",
                    "content": "Local inference runs on LM Studio (http://localhost:1234/v1) with OpenAI-compatible endpoint.",
                },
                {
                    "id": str(uuid.uuid4())[:8],
                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "category": "project",
                    "content": "Primary project workspace is F:\\Personal (Electron + Vite + Tailwind + Python zeta_bridge).",
                },
                {
                    "id": str(uuid.uuid4())[:8],
                    "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "category": "policy",
                    "content": "Autonomous IDE permissions are active (policy.allow_all) for viewing, editing, and terminal commands.",
                },
            ]
            self._write_json(self.memories_file, default_memories)

        if not os.path.exists(self.checkpoints_file):
            self._write_json(self.checkpoints_file, [])

    def _read_json(self, path: str) -> list[dict[str, Any]]:
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        return data
        except Exception as e:
            logger.error("Failed to read JSON from %s: %s", path, e)
        return []

    def _write_json(self, path: str, data: Any) -> None:
        try:
            temp_path = f"{path}.tmp"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            os.replace(temp_path, path)
        except Exception as e:
            logger.error("Failed to write JSON to %s: %s", path, e)

    # --- Core Memories Operations ---

    def get_memories(self) -> list[dict[str, Any]]:
        """Returns all saved core memory items."""
        return self._read_json(self.memories_file)

    def add_memory(self, content: str, category: str = "general") -> dict[str, Any]:
        """Saves a new persistent memory into the brain."""
        memories = self.get_memories()
        memory_item = {
            "id": str(uuid.uuid4())[:8],
            "created_at": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "category": category.strip().lower(),
            "content": content.strip(),
        }
        memories.insert(0, memory_item)
        self._write_json(self.memories_file, memories)
        logger.info("Saved new core memory [%s]: %s", memory_item["id"], content)
        return memory_item

    def delete_memory(self, memory_id: str) -> bool:
        """Removes a core memory item by its ID."""
        memories = self.get_memories()
        filtered = [m for m in memories if m.get("id") != memory_id]
        if len(filtered) != len(memories):
            self._write_json(self.memories_file, filtered)
            return True
        return False

    def format_core_memories_prompt(self) -> str:
        """Formats core memories into a system prompt section for the agent."""
        memories = self.get_memories()
        if not memories:
            return ""

        lines = [
            "\n[BRAIN CORE MEMORIES & PERSISTENT KNOWLEDGE]",
            "The following facts and project decisions are remembered in your persistent brain memory:",
        ]
        for m in memories:
            cat = m.get("category", "fact").upper()
            lines.append(f"- [{cat}] {m.get('content', '')}")
        lines.append(
            "Use these core memories to guide answers and remain consistent with established decisions.\n"
        )
        return "\n".join(lines)

    # --- Backtracking Checkpoints Operations ---

    def get_checkpoints(self) -> list[dict[str, Any]]:
        """Returns all saved conversation checkpoints."""
        return self._read_json(self.checkpoints_file)

    def create_checkpoint(
        self,
        label: str,
        messages: list[dict[str, Any]],
        auto: bool = False,
    ) -> dict[str, Any]:
        """Saves a snapshot of current conversation history for backtracking."""
        checkpoints = self.get_checkpoints()
        now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        preview = "Empty session"
        if messages:
            last_msg = messages[-1].get("content", "")
            preview = (last_msg[:80] + "...") if len(last_msg) > 80 else last_msg

        checkpoint = {
            "id": f"chk_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_{str(uuid.uuid4())[:4]}",
            "created_at": now_str,
            "label": label.strip() or f"Checkpoint ({now_str})",
            "message_count": len(messages),
            "preview": preview,
            "auto": auto,
            "messages": messages,
        }

        # Keep at most 30 checkpoints
        checkpoints.insert(0, checkpoint)
        if len(checkpoints) > 30:
            checkpoints = checkpoints[:30]

        self._write_json(self.checkpoints_file, checkpoints)
        logger.info("Created checkpoint '%s' (%d messages)", checkpoint["id"], len(messages))
        return checkpoint

    def restore_checkpoint(self, checkpoint_id: str) -> Optional[list[dict[str, Any]]]:
        """Retrieves messages from a checkpoint to backtrack the conversation."""
        checkpoints = self.get_checkpoints()
        for chk in checkpoints:
            if chk.get("id") == checkpoint_id:
                logger.info("Restoring conversation from checkpoint '%s'", checkpoint_id)
                # Return deep copy of messages
                return json.loads(json.dumps(chk.get("messages", [])))
        return None

    def delete_checkpoint(self, checkpoint_id: str) -> bool:
        """Deletes a saved checkpoint."""
        checkpoints = self.get_checkpoints()
        filtered = [c for c in checkpoints if c.get("id") != checkpoint_id]
        if len(filtered) != len(checkpoints):
            self._write_json(self.checkpoints_file, filtered)
            return True
        return False
