from pathlib import Path

from pydantic import BaseModel, Field

from pp.domain import ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool
from pp.utils.paths import resolve_path_safe


class GlobParams(BaseModel):
    pattern: str = Field(..., description="Glob pattern to match (e.g. *.py, **/*.ts)")
    path: str = Field(".", description="File or directory to search in. Defaults to current directory")


class GlobTool(Tool):
    name = "glob"
    description = "Match file paths using glob patterns"
    type = ToolType.READ
    schema = GlobParams

    MAX_RESULTS = 200
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
        params = GlobParams(**invocation.params)
        pattern = params.pattern

        try:
            path = resolve_path_safe(invocation.cwd, params.path)
        except ValueError as e:
            return ToolResult.error_result(str(e))

        if not path.exists():
            return ToolResult.error_result(f"Path does not exist: {path}")

        matches = self._glob(path, pattern)

        if not matches:
            return ToolResult.success_result(
                f"No matches found for pattern {params.pattern}",
                metadata={"matches": 0, "path": str(path)},
            )

        truncated = False

        truncated = False
        if len(matches) > self.MAX_RESULTS:
            lines = matches[: self.MAX_RESULTS]
            truncated = True

        relative = [str(p.relative_to(invocation.cwd) for p in matches)]

        return ToolResult.success_result(
            "\n".join(relative),
            metadata={"matches": len(relative), "truncated": truncated, "path": str(path)},
        )

    # ======= Helpers ======= #

    def _glob(self, path: Path, pattern: str) -> list[Path]:
        matches = []

        for match in path.glob(pattern):
            if self._is_denied(match):
                continue
            if match.is_file():
                matches.append(match)

        return matches

    def _is_denied(self, path: Path) -> bool:
        for part in path.parts:
            if part in self.DENY_DIRS:
                return True
        return False
