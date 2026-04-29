from pydantic import BaseModel, Field

from pp.domain import ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool
from pp.utils.paths import resolve_path


class WriteFileParams(BaseModel):
    path: str = Field(..., description="Path to file (relative to cwd or absolute)")
    content: str = Field(..., description="Content to be written in the file")
    create_dirs: bool = Field(True, description="Create parent directories if they don't exist. Defaults to True.")


class WriteFileTool(Tool):
    name = "write_file"
    description = (
        "Write content to a file. Creates the file if it doesn't exist, or overwrites if it does. "
        "Parent directories are created automatically. Use this for creating new files or completely replacing file contents."
        "\nFor partial modifications, use the edit tool instead."
    )
    type = ToolType.WRITE
    schema = WriteFileParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = WriteFileParams(**invocation.params)
        path = resolve_path(invocation.cwd, params.path)

        is_new_file = not path.exists()
