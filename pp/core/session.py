import json
import uuid
from datetime import datetime, timezone
from typing import Any

from pp.config import Config
from pp.context.manager import ContextManager
from pp.domain import Message, TokenUsage
from pp.hooks import HookSystem
from pp.llm import OpenRouterLLM
from pp.safety.approval import ApprovalManager
from pp.tools.mcp.manager import MCPManager
from pp.tools.registry import create_default_registry
from pp.utils.paths import get_data_dir


class Session:
    MEMORY_FILE = "memory.json"  # stored in user data directory

    def __init__(self, config: Config):
        self.config = config
        self.llm = OpenRouterLLM(config)
        self.tool_registry = create_default_registry(config)
        self.tool_registry.session = self
        self.mcp_manager = MCPManager(config)
        self.approval_manager = ApprovalManager(config)
        self.context_manager = ContextManager(
            config=config,
            user_memory=self._load_memory(),
            tools=self.tool_registry.list_tools(),
        )
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self._turn_count = 0
        self.hook_system = HookSystem(config)
        self.token_usage = TokenUsage()
        self.is_subagent = False

        # Add session reference to all tools
        for tool in self.tool_registry.list_tools():
            tool.session = self

    async def initialize(self) -> None:
        await self.mcp_manager.initialize()
        mcp_tools = await self.mcp_manager.get_tools()
        for tool in mcp_tools:
            self.tool_registry.register(tool)
        self.context_manager.update_system_prompt(self.tool_registry.list_tools())

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

        serialized_messages = []
        for msg in self.context_manager._messages:
            serialized_messages.append(
                {
                    "role": msg.role,
                    "content": msg.content,
                    "token_count": msg.token_count,
                    "tool_call_id": msg.tool_call_id,
                    "tool_calls": msg.tool_calls,
                }
            )

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
            "messages": serialized_messages,
        }

        path = sessions_dir / f"{self.session_id}.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

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

        messages = []
        for msg_data in data.get("messages", []):
            msg = Message(
                role=msg_data["role"],
                content=msg_data["content"],
                token_count=msg_data.get("token_count"),
                tool_call_id=msg_data.get("tool_call_id"),
                tool_calls=msg_data.get("tool_calls", []),
            )
            messages.append(msg)
        self.context_manager._messages = messages

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
                sessions.append(data)
        except Exception:
            pass

    sessions.sort(key=lambda s: s.get("updated_at", ""), reverse=True)
    return sessions
