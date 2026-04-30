import uuid
from datetime import datetime, timezone

from pp.config import Config
from pp.context.manager import ContextManager
from pp.llm import OpenRouterLLM
from pp.tools.registry import create_default_registry


class Session:
    def __init__(self, config: Config):
        self.config = config
        self.llm = OpenRouterLLM(config)
        self.context_manager = ContextManager(config)
        self.tool_registry = create_default_registry()
        self.session_id = str(uuid.uuid4())
        self.created_at = datetime.now(timezone.utc)
        self.updated_at = datetime.now(timezone.utc)
        self._turn_count = 0

    def increment_turn(self) -> int:
        self._turn_count += 1
        self.updated_at = datetime.now(timezone.utc)
        return self._turn_count
