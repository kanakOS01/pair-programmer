import fnmatch
import os
import re
from pathlib import Path

from pydantic import BaseModel, Field

from pp.domain import ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool
from pp.utils.paths import resolve_path_safe


class GrepParams(BaseModel):
    pattern: str = Field(..., description="Regex pattern to search for")
    path: str = Field(".", description="File or directory to search in. Defaults to current directory")
    case_insensitive: bool = Field(False, description="Case insensitive search. Defaults to False")


class GrepTool(Tool):
    name = "grep"
    description = "Search for a regex pattern in file contents. Returns matching lines with file paths and line numbers."
    type = ToolType.READ
    schema = GrepParams

    MAX_PATTERN_LENGTH = 500
    MAX_RESULTS = 200
    MAX_FILESIZE_BYTES = 1024 * 1024  # 1MB

    DENY_GLOBS = [
        "*.lock",
        "*.log",
        "*.sqlite",
        "*.db",
        "*.env",
    ]
    DENY_DIRS = {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        "venv",
        "dist",
        "build",
    }

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = GrepParams(**invocation.params)

        try:
            path = resolve_path_safe(invocation.cwd, params.path)
        except ValueError as e:
            return ToolResult.error_result(str(e))

        if not path.exists():
            return ToolResult.error_result(f"Path does not exist: {path}")

        if not params.pattern or len(params.pattern) > self.MAX_PATTERN_LENGTH:
            return ToolResult.error_result("Invalid or too long regex pattern")

        try:
            flags = re.IGNORECASE if params.case_insensitive else 0
            regex = re.compile(params.pattern, flags)
        except re.error as e:
            return ToolResult.error_result(f"Invalid regex pattern: {e}")

        files_to_search = []
        if path.is_file():
            files_to_search.append(path)
        else:
            for root, dirs, files in os.walk(path):
                # Modify dirs in-place to skip denied directories
                dirs[:] = [d for d in dirs if d not in self.DENY_DIRS]

                for file in files:
                    if any(fnmatch.fnmatch(file, glob) for glob in self.DENY_GLOBS):
                        continue
                    files_to_search.append(Path(root) / file)

        lines = []
        truncated = False

        for file_path in files_to_search:
            try:
                if file_path.stat().st_size > self.MAX_FILESIZE_BYTES:
                    continue
            except OSError:
                continue

            try:
                try:
                    rel_path = file_path.relative_to(invocation.cwd)
                except ValueError:
                    rel_path = file_path

                prefix = f"{rel_path}:"

                with open(file_path, "r", encoding="utf-8") as f:
                    for line_idx, line in enumerate(f, 1):
                        if regex.search(line):
                            lines.append(f"{prefix}{line_idx}:{line.rstrip('\n')}")
                            if len(lines) >= self.MAX_RESULTS:
                                truncated = True
                                break
            except (UnicodeDecodeError, OSError):
                # Skip binary files or files that can't be read
                pass

            if truncated:
                break

        if not lines:
            return ToolResult.success_result(
                f"No matches found for pattern {params.pattern}",
                metadata={"matches": 0, "path": str(path)},
            )

        matches_count = len(lines)
        if truncated:
            lines.append("\n... [truncated]")

        return ToolResult.success_result(
            "\n".join(lines),
            metadata={"matches": matches_count, "truncated": truncated, "path": str(path)},
        )
