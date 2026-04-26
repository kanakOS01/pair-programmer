from pydantic import BaseModel, Field

from pp.domain import ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool
from pp.utils.paths import is_binary_file, resolve_path
from pp.utils.text import count_tokens, truncate_text


class ReadFileParams(BaseModel):
    path: str = Field(..., description="Path to file (relative to cwd or absolute)")
    offset: int = Field(1, ge=1, description="Line number offset (1 based). Defaults to 1")
    limit: int | None = Field(None, ge=1, description="Maximum number of lines to read. Reads entire file if not specified.")


class ReadFileTool(Tool):
    name = "read_file"
    description = (
        "Read the contents of a text file. Returns the file content with line numbers. "
        "For large files, use offset and limit to read specific portions. "
        "Cannot read binary files (images, executables, etc.)."
    )
    type = ToolType.READ
    schema = ReadFileParams

    MAX_FILE_SIZE_BYTES = 1024 * 1024 * 10  # 10 mb
    ONE_MEGA_BYTE = 1024 * 1024
    MAX_OUTPUT_TOKENS = 25_000

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ReadFileParams(**invocation.params)
        path = resolve_path(invocation.cwd, params.path)

        if not path.exists():
            return ToolResult.error_result(f"File not found: {path}")

        if not path.is_file():
            return ToolResult.error_result(f"Path is not a file: {path}")

        file_size = path.stat().st_size
        if file_size > self.MAX_FILE_SIZE_BYTES:
            return ToolResult.error_result(
                f"File too large ({file_size / self.ONE_MEGA_BYTE:.1f} MB) "
                f"Maximum is ({self.MAX_FILE_SIZE_BYTES / self.ONE_MEGA_BYTE:.1f} MB)"
            )

        if is_binary_file(path):
            return ToolResult.error_result(f"Cannot read binary file: {path}")

        try:
            try:
                content = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                content = path.read_text(encoding="latin-1")

            lines = content.splitlines()
            total_lines = len(lines)

            if total_lines == 0:
                return ToolResult.success_result("File is empty.", metadata={"lines": 0})

            start_idx = max(0, params.offset - 1)
            end_idx = min(start_idx + params.limit, total_lines) if params.limit is not None else total_lines

            lines = lines[start_idx:end_idx]

            formatted_lines = []
            for i, line in enumerate(lines, start=start_idx + 1):
                formatted_lines.append(f"{i:7} | {line}")

            output = "\n".join(formatted_lines)

            truncated = False
            if count_tokens(output, "") > self.MAX_OUTPUT_TOKENS:
                output = truncate_text(
                    output,
                    "",
                    self.MAX_OUTPUT_TOKENS,
                )
                truncated = True

            lines_metadata = []
            if start_idx > 0 or end_idx < total_lines:
                lines_metadata.append(f"Read lines {start_idx + 1}-{end_idx}")

            if lines_metadata:
                header = " | ".join(lines_metadata) + "\n"
                output = header + output

            return ToolResult.success_result(
                output,
                truncated=truncated,
                metadata={
                    "path": str(path),
                    "total_lines": total_lines,
                    "start_line": start_idx + 1,
                    "end_line": end_idx,
                },
            )

        except Exception as e:
            return ToolResult.error_result(f"Failed to read file {path}: {e}")
