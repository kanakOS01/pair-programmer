from pp.tools.base import Tool
from pp.tools.builtin.apply_patch import ApplyPatchTool
from pp.tools.builtin.edit_file import EditFileTool
from pp.tools.builtin.glob import GlobTool
from pp.tools.builtin.grep import GrepTool
from pp.tools.builtin.list_dir import ListDirTool
from pp.tools.builtin.read_file import ReadFileTool
from pp.tools.builtin.read_url import ReadUrlTool
from pp.tools.builtin.shell import ShellTool
from pp.tools.builtin.todo import TodoTool
from pp.tools.builtin.web_search import WebSearchTool
from pp.tools.builtin.write_file import WriteFileTool

__all__ = [
    "ReadFileTool",
    "ListDirTool",
    "GrepTool",
    "GlobTool",
    "ReadUrlTool",
    "WebSearchTool",
    "WriteFileTool",
    "EditFileTool",
    "ApplyPatchTool",
    "ShellTool",
    "TodoTool",
]


def get_builtin_tools() -> list[type(Tool)]:
    return [
        ReadFileTool,
        ListDirTool,
        GrepTool,
        GlobTool,
        ReadUrlTool,
        WebSearchTool,
        WriteFileTool,
        EditFileTool,
        ApplyPatchTool,
        ShellTool,
        TodoTool,
    ]
