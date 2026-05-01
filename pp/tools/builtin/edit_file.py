from pathlib import Path

from pydantic import BaseModel, Field

from pp.domain import FileDiff, ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool
from pp.utils.paths import create_parent_dir, resolve_path


class EditFileParams(BaseModel):
    path: str = Field(..., description="Path to file (relative to cwd or absolute)")
    old_str: str = Field(
        "",
        description=(
            "The exact text to find and replace. Must match exactly including all "
            "whitespace and indentation. For new files, leave this empty."
        ),
    )
    new_str: str = Field("", description="The text to replace old_str with. Can be empty to delete old_str.")
    replace_all: bool = Field(False, description="Replace all occurrences of old_str. Defaults to false")


class EditFileTool(Tool):
    name = "edit_file"
    description = (
        "Edit a file by replacing text. The old_string must match exactly "
        "(including whitespace and indentation) and must be unique in the file "
        "unless replace_all is true. Use this for precise edits. "
        "For creating new files or complete rewrites, use write_file instead."
    )
    type = ToolType.WRITE
    schema = EditFileParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = EditFileParams(**invocation.params)
        path = resolve_path(invocation.cwd, params.path)

        if not path.exists():
            if params.old_str:
                return ToolResult.error_result(f"{path} file does not exist. Use write_file tool instead")

            create_parent_dir(path)
            path.write_text(params.new_str, encoding="utf-8")
            return ToolResult.success_result(
                f"Created {path}",
                diff=FileDiff(path=path, old="", new=params.new_str, is_new_file=True),
                metadata={
                    "path": str(path),
                    "is_new_file": True,
                    "lines": len(params.new_str.splitlines()),
                    "bytes": len(params.new_str.encode("utf-8")),
                },
            )

        if not params.old_str:
            return ToolResult.error_result("old_str is empty but file exists. Use old_str to edit or write_file tool instead")

        old = ""
        try:
            old = path.read_text(encoding="utf-8")
        except Exception as e:
            # failed to read old content, but we can still write the new content
            pass

        occurences = old.count(params.old_str)
        if occurences == 0:
            return ToolResult.error_result(f"old_str not found in {path}")

        if occurences > 1 and not params.replace_all:
            return ToolResult.error_result(
                f"old_str found {occurences} times. Use replace_all=true to replace all or provide more specific old_str",
                metadata={
                    "occurences": occurences,
                },
            )

        if params.replace_all:
            new = old.replace(params.old_str, params.new_str)
        else:
            new = old.replace(params.old_str, params.new_str, 1)

        if new == old:
            return ToolResult.error_result("No changes made to the file")

        try:
            path.write_text(new, encoding="utf-8")
        except IOError as e:
            return ToolResult.error_result(f"Failed to write to {path}: {e}")

        return ToolResult.success_result(
            f"Edited {path}: replaced {occurences} instance(s)",
            diff=FileDiff(path=path, old=old, new=new, is_new_file=False),
            metadata={
                "path": str(path),
                "is_new_file": False,
                "lines": len(new.splitlines()),
                "bytes": len(new.encode("utf-8")),
            },
        )

    def _no_match_error(self, old_string: str, content: str, path: Path) -> ToolResult:
        lines = content.splitlines()

        partial_matches = []
        search_terms = old_string.split()[:5]

        if search_terms:
            first_term = search_terms[0]
            for i, line in enumerate(lines, 1):
                if first_term in line:
                    partial_matches.append((i, line.strip()[:80]))
                    if len(partial_matches) >= 3:
                        break

        error_msg = f"old_string not found in {path}."

        if partial_matches:
            error_msg += "\n\nPossible similar lines:"
            for line_num, line_preview in partial_matches:
                error_msg += f"\n  Line {line_num}: {line_preview}"
            error_msg += "\n\nMake sure old_string matches exactly (including whitespace and indentation)."
        else:
            error_msg += (
                " Make sure the text matches exactly, including:\n"
                "- All whitespace and indentation\n"
                "- Line breaks\n"
                "- Any invisible characters\n"
                "Try re-reading the file using read_file tool and then editing."
            )

        return ToolResult.error_result(error_msg)
