"""内置工具

框架自带的工具实现，包括任务管理工具 todo_write。
"""

from dataclasses import dataclass
from typing import Any

from .tool import Tool

# 合法的任务状态值
VALID_STATUSES = ("pending", "in_progress", "completed", "cancelled")

# 合法的优先级值
VALID_PRIORITIES = ("high", "medium", "low")


@dataclass
class TodoItem:
    """单个任务项

    Attributes:
        content: 任务内容描述
        status: 任务状态 (pending/in_progress/completed/cancelled)
        priority: 优先级 (high/medium/low)
    """

    content: str
    status: str = "pending"
    priority: str = "medium"

    def __post_init__(self):
        """校验状态值和优先级合法性"""
        if self.status not in VALID_STATUSES:
            raise ValueError(
                f"无效的任务状态: '{self.status}'，"
                f"仅支持: {', '.join(VALID_STATUSES)}"
            )
        if self.priority not in VALID_PRIORITIES:
            raise ValueError(
                f"无效的优先级: '{self.priority}'，"
                f"仅支持: {', '.join(VALID_PRIORITIES)}"
            )

    def to_dict(self) -> dict[str, str]:
        """转换为字典"""
        return {"content": self.content, "status": self.status, "priority": self.priority}


class TodoStore:
    """任务列表内存存储

    管理 Agent 运行期间的 todo 状态。每次 write() 调用完全替换列表。
    """

    def __init__(self):
        self._todos: list[TodoItem] = []

    def write(self, todos: list[dict[str, str]]) -> str:
        """替换整个 todo 列表

        Args:
            todos: 新的任务列表，每项包含 content、status，可选 priority

        Returns:
            格式化的当前 todo 列表字符串（注入 LLM 上下文）
        """
        try:
            self._todos = [TodoItem(**t) for t in todos]
        except ValueError as e:
            return f"❌ 任务更新失败: {e}"
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
            "in_progress": "🔵",
            "completed": "✅",
            "cancelled": "⛔",
        }

        lines = ["📋 当前任务列表："]
        for i, todo in enumerate(self._todos, 1):
            icon = status_icons.get(todo.status, "❓")
            lines.append(f"  {i}. {icon} [{todo.status}] {todo.content}")

        pending = sum(1 for t in self._todos if t.status == "pending")
        in_progress = sum(1 for t in self._todos if t.status == "in_progress")
        completed = sum(1 for t in self._todos if t.status == "completed")
        cancelled = sum(1 for t in self._todos if t.status == "cancelled")
        lines.append(
            f"\n总计: {len(self._todos)} 项 | "
            f"待处理: {pending} | 进行中: {in_progress} | "
            f"已完成: {completed} | 已取消: {cancelled}"
        )
        return "\n".join(lines)


# todo_write 工具的详细描述（参考 opencode todowrite.txt 翻译并适配通用场景）
TODO_WRITE_DESCRIPTION = """任务管理工具 - 创建和管理任务列表，追踪多步骤任务的执行进度。

## 何时使用
- 需要 3 个或更多步骤的复杂任务
- 用户一次提供了多个待办事项
- 需要规划和追踪的多阶段工作
- 长时间运行的任务，需要展示进度

## 何时不使用
- 单步的简单任务（如回答一个问题、执行一次搜索）
- 纯对话交流
- 不需要进度追踪的即时操作

## 任务状态说明
- pending: 待处理，尚未开始
- in_progress: 进行中，正在执行
- completed: 已完成
- cancelled: 已取消，不再需要

## 优先级说明
- high: 高优先级，优先处理
- medium: 中优先级（默认）
- low: 低优先级，可延后处理

## 管理规则
- 每次调用会完全替换当前任务列表
- 同一时间只保持一个任务为 in_progress 状态
- 完成一个步骤后立即更新状态为 completed
- 开始下一步前将其标记为 in_progress
- 按优先级和逻辑顺序排列任务"""


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
            todos: 任务列表，每项包含 content、status，可选 priority
        """
        return store.write(todos)

    return Tool(
        name="todo_write",
        description=TODO_WRITE_DESCRIPTION,
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
                                "description": "任务内容描述",
                            },
                            "status": {
                                "type": "string",
                                "enum": list(VALID_STATUSES),
                                "description": "任务状态: pending(待处理)/in_progress(进行中)/completed(已完成)/cancelled(已取消)",
                            },
                            "priority": {
                                "type": "string",
                                "enum": list(VALID_PRIORITIES),
                                "description": "优先级: high(高)/medium(中，默认)/low(低)",
                                "default": "medium",
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
