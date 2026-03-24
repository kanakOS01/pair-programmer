import subprocess
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
    description = (
        "Search for a regex pattern in file contents. Returns matching lines with file paths and line numbers. "
        "Uses ripgrep (rg), sandboxed to workspace."
    )
    type = ToolType.READ
    schema = GrepParams

    MAX_PATTERN_LENGTH = 500
    MAX_RESULTS = 200
    MAX_FILESIZE_STR = "1M"
    TIMEOUT_SECONDS = 2

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
            return ToolResult.error_result("Invalid or too long regext pattern")

        cmd = self._build_rg_command(params.pattern, path, params.case_insensitive)

        try:
            result = subprocess.run(
                cmd,
                cwd=invocation.cwd,
                capture_output=True,
                text=True,
                timeout=self.TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired:
            return ToolResult.error_result("Search timed out")

        # rg exit codes:
        # 0 = match found
        # 1 = no matches
        # 2 = error
        if result.returncode == 2:
            return ToolResult.error_result(result.stderr.strip() or "ripgrep internal error")

        output = result.stdout.strip()

        if not output:
            return ToolResult.success_result(
                f"No matches found for pattern {params.pattern}",
                metadata={"matches": 0, "path": str(path)},
            )

        lines = output.splitlines()

        truncated = False
        if len(lines) > self.MAX_RESULTS:
            lines = lines[: self.MAX_RESULTS]
            truncated = True

        return ToolResult.success_result(
            "\n".join(lines),
            metadata={"matches": len(lines), "truncated": truncated, "path": str(path)},
        )

    # ======= Helpers ======= #

    def _build_rg_command(self, pattern: str, path: Path, case_insensitive: bool) -> list[str]:
        cmd = [
            "rg",
            "--line-number",
            "--no-heading",
            "--color",
            "never",
            "--max-count",
            str(self.MAX_RESULTS),
            "--max-filesize",
            self.MAX_FILESIZE_STR,
        ]

        if case_insensitive:
            cmd.append("--ignore-case")

        # exclude directories
        for d in self.DENY_DIRS:
            cmd.extend(["--glob", f"!{d}/**"])

        # exclude file patterns
        for g in self.DENY_GLOBS:
            cmd.extend(["--glob", f"!{g}"])

        cmd.extend([pattern, str(path)])
        return cmd
