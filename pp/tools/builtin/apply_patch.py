from typing import List

from pydantic import BaseModel, Field

from pp.domain import FileDiff, ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool
from pp.utils.paths import resolve_path


class PatchChunk(BaseModel):
    old_str: str = Field(
        ...,
        description=("The exact text to find and replace. Must match exactly including all whitespace and indentation."),
    )
    new_str: str = Field(..., description="The text to replace old_str with. Can be empty to delete old_str.")
    replace_all: bool = Field(False, description="Replace all occurrences of old_str. Defaults to false")


class ApplyPatchParams(BaseModel):
    path: str = Field(..., description="Path to file (relative to cwd or absolute)")
    patches: List[PatchChunk] = Field(..., description="List of patches/replacements to apply to the file.")


class ApplyPatchTool(Tool):
    name = "apply_patch"
    description = (
        "Edit a file by applying multiple text replacements (patches). Each patch's "
        "old_string must match exactly (including whitespace and indentation). "
        "Use this for making multiple non-contiguous precise edits in a single file."
    )
    type = ToolType.WRITE
    schema = ApplyPatchParams

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = ApplyPatchParams(**invocation.params)
        path = resolve_path(invocation.cwd, params.path)

        if not path.exists():
            return ToolResult.error_result(f"{path} file does not exist. Use write_file tool instead for new files.")

        try:
            old = path.read_text(encoding="utf-8")
        except Exception as e:
            return ToolResult.error_result(f"Failed to read file {path}: {e}")

        new = old
        applied_count = 0

        for i, patch in enumerate(params.patches):
            if not patch.old_str:
                return ToolResult.error_result(f"Patch at index {i} has an empty old_str. Use old_str to match text.")

            occurences = new.count(patch.old_str)
            if occurences == 0:
                return ToolResult.error_result(f"old_str from patch {i} not found in the current state of {path}")

            if occurences > 1 and not patch.replace_all:
                return ToolResult.error_result(
                    f"old_str from patch {i} found {occurences} times. "
                    "Use replace_all=true to replace all or provide more specific old_str",
                    metadata={
                        "occurences": occurences,
                        "patch_index": i,
                    },
                )

            if patch.replace_all:
                new = new.replace(patch.old_str, patch.new_str)
            else:
                new = new.replace(patch.old_str, patch.new_str, 1)

            applied_count += 1

        if new == old:
            return ToolResult.error_result("No changes were made to the file after applying all patches")

        try:
            path.write_text(new, encoding="utf-8")
        except IOError as e:
            return ToolResult.error_result(f"Failed to write to {path}: {e}")

        return ToolResult.success_result(
            f"Edited {path}: successfully applied {applied_count} patch(es)",
            diff=FileDiff(path=path, old=old, new=new, is_new_file=False),
            metadata={
                "path": str(path),
                "is_new_file": False,
                "lines": len(new.splitlines()),
                "bytes": len(new.encode("utf-8")),
                "patches_applied": applied_count,
            },
        )
