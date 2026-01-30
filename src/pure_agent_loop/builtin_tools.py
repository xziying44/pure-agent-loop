"""内置工具

框架自带的工具实现，包括任务管理工具 todo_write。
"""

from dataclasses import dataclass
from typing import Any

from .tool import Tool


@dataclass
class TodoItem:
    """单个任务项

    Attributes:
        content: 任务内容描述
        status: 任务状态 (pending/in_progress/completed)
    """

    content: str
    status: str = "pending"

    def to_dict(self) -> dict[str, str]:
        """转换为字典"""
        return {"content": self.content, "status": self.status}


class TodoStore:
    """任务列表内存存储

    管理 Agent 运行期间的 todo 状态。每次 write() 调用完全替换列表。
    """

    def __init__(self):
        self._todos: list[TodoItem] = []

    def write(self, todos: list[dict[str, str]]) -> str:
        """替换整个 todo 列表

        Args:
            todos: 新的任务列表，每项包含 content 和 status

        Returns:
            格式化的当前 todo 列表字符串（注入 LLM 上下文）
        """
        self._todos = [TodoItem(**t) for t in todos]
        return self._format_output()

    @property
    def todos(self) -> list[TodoItem]:
        """获取当前 todo 列表（返回副本）"""
        return list(self._todos)

    def _format_output(self) -> str:
        """格式化当前 todo 列表"""
        if not self._todos:
            return "📋 任务列表为空"

        status_icons = {
            "pending": "⬜",
            "in_progress": "🔄",
            "completed": "✅",
        }

        lines = ["📋 当前任务列表："]
        for i, todo in enumerate(self._todos, 1):
            icon = status_icons.get(todo.status, "❓")
            lines.append(f"  {i}. {icon} [{todo.status}] {todo.content}")

        pending = sum(1 for t in self._todos if t.status == "pending")
        in_progress = sum(1 for t in self._todos if t.status == "in_progress")
        completed = sum(1 for t in self._todos if t.status == "completed")
        lines.append(
            f"\n总计: {len(self._todos)} 项 | "
            f"待处理: {pending} | 进行中: {in_progress} | 已完成: {completed}"
        )
        return "\n".join(lines)


def create_todo_tool(store: TodoStore) -> Tool:
    """创建绑定到指定 TodoStore 的 todo_write 工具

    Args:
        store: TodoStore 实例，工具执行时操作此 store

    Returns:
        Tool 实例
    """

    def todo_write(todos: list[dict[str, str]]) -> str:
        """更新任务列表，完全替换当前列表

        Args:
            todos: 任务列表，每项包含 content（任务内容）和 status（pending/in_progress/completed）
        """
        return store.write(todos)

    return Tool(
        name="todo_write",
        description="更新任务列表，完全替换当前列表。每个任务项包含 content（任务内容）和 status（pending/in_progress/completed）。",
        parameters={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "任务列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "任务内容",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "任务状态",
                            },
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["todos"],
        },
        function=todo_write,
        is_async=False,
    )
