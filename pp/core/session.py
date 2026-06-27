import json
import uuid
from datetime import datetime, timezone

from pp.config import Config
from pp.context.manager import ContextManager
from pp.llm import OpenRouterLLM
from pp.tools.registry import create_default_registry
from pp.utils.paths import get_data_dir


class Session:
    MEMORY_FILE = "memory.json"  # stored in user data directory

    def __init__(self, config: Config):
        self.config = config
        self.llm = OpenRouterLLM(config)
        self.tool_registry = create_default_registry(config)
        self.context_manager = ContextManager(
            config=config,
            user_memory=self._load_memory(),
            tools=self.tool_registry.list_tools(),
        )
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self._turn_count = 0

    def increment_turn(self) -> int:
        self._turn_count += 1
        self.updated_at = datetime.now(timezone.utc)
        return self._turn_count

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
