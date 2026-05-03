import uuid
from typing import Any

from pydantic import BaseModel, Field

from pp.config import Config
from pp.domain import TodoAction, ToolInvocation, ToolResult, ToolType
from pp.tools.base import Tool


class TodoParams(BaseModel):
    action: str = Field(..., description="Action to perform (add/bulk_add/done/list/clear)")
    message: str | None = Field(None, description="Todo content (for add action)")
    messages: list[str] | None = Field(None, description="List of todo contents (for bulk_add action)")
    id: str | None = Field(None, description="Todo ID (for done action or add (optional: will generate random id if empty))")


class TodoTool(Tool):
    name = "todos"
    description = "Manage task list for the current session. Use this to track progress for multi step tasks."
    type = ToolType.MEMORY
    schema = TodoParams

    def __init__(self, config: Config):
        super().__init__(config)
        self._todos: dict[str, dict[str, Any]] = {}

    async def execute(self, invocation: ToolInvocation) -> ToolResult:
        params = TodoParams(**invocation.params)

        if params.action.lower() == TodoAction.ADD:
            if not params.message:
                return ToolResult.error_result("`message` is required for `add` action")
            todo_id = params.id or str(uuid.uuid4())
            self._todos[todo_id] = {
                "message": params.message,
                "status": "pending",
            }
            return ToolResult.success_result(f"Todo added [{todo_id}] (pending) {params.message}")

        elif params.action.lower() == TodoAction.BULK_ADD:
            if not params.messages:
                return ToolResult.error_result("`messages` is required for `bulk_add` action")

            added_ids = []
            for msg in params.messages:
                todo_id = str(uuid.uuid4())
                self._todos[todo_id] = {
                    "message": msg,
                    "status": "pending",
                }
                added_ids.append(todo_id)

            return ToolResult.success_result(f"Bulk added {len(added_ids)} todos successfully")

        elif params.action.lower() == TodoAction.DONE:
            if not params.id:
                return ToolResult.error_result("`id` is required for `done` action")
            if params.id not in self._todos:
                return ToolResult.error_result("Todo not found")
            self._todos[params.id]["status"] = "done"
            return ToolResult.success_result(f"Todo done [{params.id}]")

        elif params.action == "list":
            if not self._todos:
                return ToolResult.success_result("No todos found", meta={"todos": []})

            todos_meta = []
            todos = []
            for tid, todo in self._todos.items():
                todos.append(f"[{tid}] ({todo['status']}) {todo['message']}")
                todos_meta.append({"id": tid, "status": todo["status"], "message": todo["message"]})

            output = "Todos:" + "\n" + "\n".join(todos)
            return ToolResult.success_result(output, meta={"todos": todos_meta})

        elif params.action == "clear":
            cnt = len(self._todos)
            self._todos.clear()
            return ToolResult.success_result(f"{cnt} todos cleared successfully")

        else:
            return ToolResult.error_result(f"Unknown action: {params.action}")
