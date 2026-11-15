import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any

from pp.config import Config
from pp.context import ConversationHistory
from pp.domain import LLMEventType, TokenUsage, UserMessage
from pp.hooks import HookSystem
from pp.llm import OpenRouterLLM
from pp.prompts.system import get_compaction_prompt, get_system_prompt
from pp.tools.base import Tool
from pp.tools.executor import ToolExecutor
from pp.tools.mcp.manager import MCPManager
from pp.tools.registry import create_default_registry
from pp.utils.paths import get_data_dir
from pp.utils.text import count_tokens


class Session:
    MEMORY_FILE = "memory.json"  # stored in user data directory

    def __init__(self, config: Config):
        self.config = config
        self.llm = OpenRouterLLM(config)
        self.tool_registry = create_default_registry(config)
        self.tool_registry.session = self
        self.mcp_manager = MCPManager(config)
        self.mcp_tools: list[Tool] = []
        self.user_memory = self._load_memory()
        self.history = ConversationHistory(count_tokens=self._count_tokens)
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self._turn_count = 0
        self.hook_system = HookSystem(config)
        # The CLI sets tool_executor.confirm. Without it, calls that need confirmation are rejected.
        self.tool_executor = ToolExecutor(self.tool_registry, config, self.hook_system)
        self.token_usage = TokenUsage()
        self.is_subagent = False

        # Add session reference to all tools
        for tool in self.tool_registry.list_tools():
            tool.session = self

    async def initialize(self) -> None:
        await self.mcp_manager.initialize()
        self.mcp_tools = list(await self.mcp_manager.get_tools())
        self._register_mcp_tools()

    def rebuild_tool_registry(self) -> None:
        """Rebuild the registry from the current config. Keep the MCP tools and the session references."""
        self.tool_registry = create_default_registry(self.config)
        self.tool_registry.session = self
        for tool in self.tool_registry.list_tools():
            tool.session = self
        self._register_mcp_tools()
        self.tool_executor.registry = self.tool_registry

    def _register_mcp_tools(self) -> None:
        for tool in self.mcp_tools:
            if self.config.allowed_tools is None or tool.name in self.config.allowed_tools:
                self.tool_registry.register(tool)

    def system_prompt(self) -> str:
        """Build the system prompt from the current config, tools and memory. It is built for each LLM request,
        so changes to the model, the approval policy or the tool list need no extra step."""
        return get_system_prompt(self.config, self.user_memory, self.tool_registry.list_tools())

    def needs_compaction(self) -> bool:
        used = self._count_tokens(self.system_prompt()) + self.history.token_count()
        return used > self.config.model.context_window * self.config.compaction_threshold

    async def compact(self, keep_last_n: int = 4) -> tuple[str, TokenUsage] | None:
        """Compact the conversation history. Return the summary and the LLM usage, or None if there was nothing to compact."""
        usage = TokenUsage()

        async def summarize(text: str) -> str:
            nonlocal usage
            summary = ""
            prompt = f"Please summarize the following conversation history concisely:\n\n{text}"
            async for event in self.llm.generate(
                [UserMessage(prompt)], tools=None, system_prompt=get_compaction_prompt(), stream=False
            ):
                if event.type == LLMEventType.TextDelta and event.text_delta:
                    summary += event.text_delta.text
                elif event.type == LLMEventType.Done:
                    if event.text_delta and not summary:
                        summary = event.text_delta.text
                    if event.usage:
                        usage = event.usage
                elif event.type == LLMEventType.Error:
                    raise RuntimeError(f"LLM compaction failed: {event.error}")
            return summary

        summary = await self.history.compact(summarize, keep_last_n)
        if summary is None:
            return None
        self.accumulate_usage(usage)
        self.save()
        return summary, usage

    def _count_tokens(self, text: str) -> int:
        return count_tokens(text, self.config.model_name)

    def increment_turn(self) -> int:
        self._turn_count += 1
        self.updated_at = datetime.now(timezone.utc)
        return self._turn_count

    def accumulate_usage(self, usage: TokenUsage) -> None:
        self.token_usage += usage
        self.updated_at = datetime.now(timezone.utc)

    def save(self) -> None:
        if getattr(self, "is_subagent", False):
            return

        sessions_dir = get_data_dir() / "sessions"
        sessions_dir.mkdir(parents=True, exist_ok=True)

        data = {
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "turn_count": self._turn_count,
            "token_usage": {
                "prompt_tokens": self.token_usage.prompt_tokens,
                "completion_tokens": self.token_usage.completion_tokens,
                "total_tokens": self.token_usage.total_tokens,
                "cached_tokens": self.token_usage.cached_tokens,
            },
            "history": self.history.to_dict(),
        }

        # Write to a temporary file, then rename, so that a crash cannot leave a corrupt session file.
        path = sessions_dir / f"{self.session_id}.json"
        tmp_path = path.with_suffix(".json.tmp")
        tmp_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        os.replace(tmp_path, path)

    def load(self, session_id: str) -> None:
        sessions_dir = get_data_dir() / "sessions"
        path = sessions_dir / f"{session_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Session {session_id} not found")

        data = json.loads(path.read_text(encoding="utf-8"))
        self.session_id = data["session_id"]
        self.created_at = datetime.fromisoformat(data["created_at"])
        self.updated_at = datetime.fromisoformat(data["updated_at"])
        self._turn_count = data.get("turn_count", 0)

        usage_data = data.get("token_usage", {})
        self.token_usage = TokenUsage(
            prompt_tokens=usage_data.get("prompt_tokens", 0),
            completion_tokens=usage_data.get("completion_tokens", 0),
            total_tokens=usage_data.get("total_tokens", 0),
            cached_tokens=usage_data.get("cached_tokens", 0),
        )

        # Old session files store a "messages" list. ConversationHistory converts it.
        history_data = data.get("history", data.get("messages", []))
        self.history = ConversationHistory.from_dict(history_data, count_tokens=self._count_tokens)

    def _load_memory(self) -> str | None:
        dir = get_data_dir()
        dir.mkdir(parents=True, exist_ok=True)
        path = dir / self.MEMORY_FILE

        if path.exists():
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                entries = data.get("entries")
                lines = ["User preferences and notes: "]
                for k, v in entries.items():
                    lines.append(f"- {k}: {v}")
                return "\n".join(lines)

            except Exception as e:
                return None
        return None


def get_sessions() -> list[dict[str, Any]]:
    sessions_dir = get_data_dir() / "sessions"
    if not sessions_dir.exists():
        return []

    sessions = []
    for file in sessions_dir.glob("*.json"):
        try:
            data = json.loads(file.read_text(encoding="utf-8"))
            if "session_id" in data:
                history_data = data.get("history", data.get("messages", []))
                data["preview"] = ConversationHistory.from_dict(history_data, count_tokens=lambda text: 0).last_text()
                sessions.append(data)
        except Exception:
            pass

    sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return sessions
