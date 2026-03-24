from pp.tools.base import Tool
from pp.tools.builtin.grep import GrepTool
from pp.tools.builtin.list_dir import ListDirTool
from pp.tools.builtin.read_file import ReadFileTool

__all__ = ["ReadFileTool", "ListDirTool", "GrepTool"]


def get_builtin_tools() -> list[type(Tool)]:
    return [ReadFileTool, ListDirTool, GrepTool]
