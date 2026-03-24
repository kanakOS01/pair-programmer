from pydantic import BaseModel, Field

from pp.domain import ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool
from pp.utils.paths import resolve_path


class ListDirParams(BaseModel):
    path: str = Field(".", description="Directory path that needs to be listed. Defaults to current directory.")
    include_hidden: bool = Field(False, description="Whether to include hidden files and directories. Defaults to False.")


class ListDirTool(Tool):
    name = "list_dir"
    description = "List the contents of a directory"
    type = ToolType.READ
    schema = ListDirParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ListDirParams(**invocation.params)
        path = resolve_path(invocation.cwd, params.path)

        if not path.exists() or not path.is_dir():
            return ToolResult.error_result(f"Path is not a directory: {path}")

        try:
            items = sorted(path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
        except Exception as e:
            return ToolResult.error_result(f"Error listing directory: {e}")

        if not params.include_hidden:
            items = [item for item in items if item.name.startswith(".")]

        if not items:
            return ToolResult.success_result("Directory is empty", metadata={"path": str(path), "entries": 0})

        item_names = [f"{item.name}/" if item.is_dir() else item.name for item in items]

        return ToolResult.success_result("\n".join(item_names), metadata={"path": str(path), "entries": len(item_names)})
