from pydantic import BaseModel, Field

from pp.domain import FileDiff, ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool
from pp.utils.paths import create_parent_dir, resolve_path


class WriteFileParams(BaseModel):
    path: str = Field(..., description="Path to file (relative to cwd or absolute)")
    content: str = Field(..., description="Content to be written in the file")
    create_dirs: bool = Field(True, description="Create parent directories if they don't exist. Defaults to true.")


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

        old = ""
        if not is_new_file:
            try:
                old = path.read_text(encoding="utf-8")
            except Exception as e:
                # failed to read old content, but we can still write the new content
                pass

        try:
            if params.create_dirs:
                create_parent_dir(path)
            elif not path.parent.exists():
                return ToolResult.error_result(f"Parent directory does not exist for file {path}")

            action = "Created" if is_new_file else "Updated"
            line_count = len(params.content.splitlines())

            path.write_text(params.content, encoding="utf-8")
            return ToolResult.success_result(
                f"{action} {line_count} lines in file {path}",
                diff=FileDiff(path=path, old=old, new=params.content, is_new_file=is_new_file),
                metadata={
                    "path": str(path),
                    "is_new_file": is_new_file,
                    "lines": line_count,
                    "bytes": len(params.content.encode("utf-8")),
                },
            )
        except OSError as e:
            return ToolResult.error_result(f"Failed to write file {path}: {e}")
