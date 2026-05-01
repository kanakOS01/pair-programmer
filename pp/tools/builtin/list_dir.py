from pydantic import BaseModel, Field

from pp.domain import ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool
from pp.utils.paths import resolve_path


class ListDirParams(BaseModel):
    path: str = Field(".", description="Directory path that needs to be listed. Defaults to current directory.")
    include_hidden: bool = Field(False, description="Whether to include hidden files and directories. Defaults to False.")
    max_depth: int = Field(0, description="Maximum depth for recursive listing. 0 means current directory only. Defaults to 0.")


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

        def _list_dir(current_path, depth, prefix=""):
            try:
                items = sorted(current_path.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower()))
            except Exception as e:
                if depth == 0:
                    raise e
                return []

            if not params.include_hidden:
                items = [item for item in items if not item.name.startswith(".")]

            result = []
            for i, item in enumerate(items):
                is_last = i == len(items) - 1
                connector = "└── " if is_last else "├── "

                name = f"{item.name}/" if item.is_dir() else item.name
                result.append(f"{prefix}{connector}{name}")

                if item.is_dir() and depth < params.max_depth:
                    child_prefix = prefix + ("    " if is_last else "│   ")
                    result.extend(_list_dir(item, depth + 1, child_prefix))
            return result

        try:
            item_names = _list_dir(path, 0)
        except Exception as e:
            return ToolResult.error_result(f"Error listing directory: {e}")

        if not item_names:
            return ToolResult.success_result("Directory is empty", metadata={"path": str(path), "entries": 0})

        return ToolResult.success_result("\n".join(item_names), metadata={"path": str(path), "entries": len(item_names)})
