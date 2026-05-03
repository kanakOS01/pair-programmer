import json

from pydantic import BaseModel, Field

from pp.domain import MemoryAction, ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool
from pp.utils.paths import get_data_dir


class MemoryParams(BaseModel):
    action: str = Field(..., description="Action to perform (set/get/delete/list/clear)")
    key: str | None = Field(None, description="Memory key (for `set`, `get`, `delete` actions)")
    value: str | None = Field(None, description="Value to store (for `set` action)")


class MemoryTool(Tool):
    name = "memory"
    description = "Manage persistent memory. Use this to store and retrieve information across sessions."
    type = ToolType.MEMORY
    schema = MemoryParams

    MEMORY_FILE = "memory.json"  # stored in user data directory

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = MemoryParams(**invocation.params)

        if params.action.lower() == MemoryAction.SET:
            if not params.key:
                return ToolResult.error_result("`key` is required for `set` action")
            if not params.value:
                return ToolResult.error_result("`value` is required for `set` action")

            mem = self._load_memory()
            mem["entries"][params.key] = params.value
            self._save_memory(mem)

            return ToolResult.success_result(f"Saved `{params.key}`")

        elif params.action.lower() == MemoryAction.GET:
            if not params.key:
                return ToolResult.error_result("`key` is required for `get` action")

            mem = self._load_memory()
            value = mem["entries"].get(params.key)
            if value is None:
                return ToolResult.error_result(f"Key `{params.key}` not found")

            return ToolResult.success_result(value, metadata={"found": True})

        elif params.action.lower() == MemoryAction.DELETE:
            if not params.key:
                return ToolResult.error_result("`key` is required for `delete` action")

            mem = self._load_memory()
            if params.key not in mem["entries"]:
                return ToolResult.error_result(f"Key `{params.key}` not found")

            del mem["entries"][params.key]
            self._save_memory(mem)

            return ToolResult.success_result(f"Deleted `{params.key}`")

        elif params.action.lower() == MemoryAction.LIST:
            mem = self._load_memory()
            if not mem["entries"]:
                return ToolResult.success_result("No memory entries found", metadata={"entries": []})

            entries_meta = []
            for key, value in mem["entries"].items():
                entries_meta.append({"key": key, "value": value})

            output = "Memory Entries:" + "\n" + "\n".join([f"[{key}] {value}" for key, value in mem["entries"].items()])
            return ToolResult.success_result(output, metadata={"entries": entries_meta, "found": True})

        elif params.action.lower() == MemoryAction.CLEAR:
            mem = self._load_memory()
            cnt = len(mem["entries"])
            mem["entries"].clear()
            self._save_memory(mem)
            return ToolResult.success_result(f"{cnt} memory entries cleared")

        else:
            return ToolResult.error_result(f"Unknown action: {params.action}")

    def _load_memory(self) -> dict[str, dict]:
        dir = get_data_dir()
        dir.mkdir(parents=True, exist_ok=True)
        path = dir / self.MEMORY_FILE

        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except Exception as e:
                return {"entries": {}}
        return {"entries": {}}

    def _save_memory(self, mem: dict[str, dict]):
        dir = get_data_dir()
        dir.mkdir(parents=True, exist_ok=True)
        path = dir / self.MEMORY_FILE

        path.write_text(json.dumps(mem, indent=2, ensure_ascii=False), encoding="utf-8")
