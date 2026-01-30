"""Rich 美化渲染器

使用 rich 库提供更美观、更清晰的事件流展示。
这是一个演示模块，展示如何自定义事件渲染。

使用前需安装: pip install rich
"""

from typing import Any

from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from pure_agent_loop import Event, EventType


class RichRenderer:
    """基于 Rich 库的美化事件渲染器

    特点：
    - 视觉层次分明：使用面板、分隔线和缩进
    - 智能折叠：长内容自动折叠显示
    - 颜色编码：不同事件类型用不同颜色
    - 表格化：Todo 列表用表格展示
    - 减少冗余：合并重复信息
    """

    # 颜色配置
    COLORS = {
        "thought": "cyan",
        "action": "yellow",
        "observation": "green",
        "error": "red",
        "soft_limit": "orange1",
        "loop": "blue",
        "todo": "magenta",
    }

    # 图标配置
    ICONS = {
        EventType.LOOP_START: "🚀",
        EventType.THOUGHT: "💭",
        EventType.ACTION: "🔧",
        EventType.OBSERVATION: "📋",
        EventType.SOFT_LIMIT: "⚠️",
        EventType.ERROR: "❌",
        EventType.LOOP_END: "✅",
        EventType.TODO_UPDATE: "📝",
    }

    def __init__(
        self,
        console: Console | None = None,
        max_thought_lines: int = 3,
        max_result_chars: int = 150,
        show_todo_table: bool = True,
        compact_mode: bool = False,
    ):
        """初始化渲染器

        Args:
            console: Rich Console 实例，可选
            max_thought_lines: 思考内容最大显示行数
            max_result_chars: 工具结果最大显示字符数
            show_todo_table: 是否用表格显示 Todo 列表
            compact_mode: 紧凑模式，减少空白行
        """
        self.console = console or Console()
        self.max_thought_lines = max_thought_lines
        self.max_result_chars = max_result_chars
        self.show_todo_table = show_todo_table
        self.compact_mode = compact_mode

        # 跟踪上一次 Todo 列表状态，避免重复显示
        self._last_todos: list[dict] | None = None

    def render(self, event: Event) -> None:
        """渲染事件到控制台

        Args:
            event: 要渲染的事件
        """
        match event.type:
            case EventType.LOOP_START:
                self._render_loop_start(event)
            case EventType.THOUGHT:
                self._render_thought(event)
            case EventType.ACTION:
                self._render_action(event)
            case EventType.OBSERVATION:
                self._render_observation(event)
            case EventType.SOFT_LIMIT:
                self._render_soft_limit(event)
            case EventType.ERROR:
                self._render_error(event)
            case EventType.LOOP_END:
                self._render_loop_end(event)
            case EventType.TODO_UPDATE:
                self._render_todo_update(event)
            case _:
                self._render_unknown(event)

    def _render_loop_start(self, event: Event) -> None:
        """渲染循环开始事件"""
        task = event.data.get("task", "")

        panel = Panel(
            Text(task, style="bold white"),
            title=f"{self.ICONS[EventType.LOOP_START]} 开始任务",
            title_align="left",
            border_style=self.COLORS["loop"],
            padding=(0, 1),
        )
        self.console.print()
        self.console.print(panel)

    def _render_thought(self, event: Event) -> None:
        """渲染思考事件（模型返回内容不折叠）"""
        content = event.data.get("content", "")

        text = Text()
        text.append(f"{self.ICONS[EventType.THOUGHT]} ", style="bold")
        text.append(content, style=self.COLORS["thought"])

        if not self.compact_mode:
            self.console.print()
        self.console.print(text)

    def _render_action(self, event: Event) -> None:
        """渲染工具调用事件"""
        tool = event.data.get("tool", "")

        # 隐藏 todo_write 工具调用，因为 TODO_UPDATE 事件已经展示了
        if tool == "todo_write":
            return

        args = event.data.get("args", {})
        args_display = self._format_args(args)

        text = Text()
        text.append(f"{self.ICONS[EventType.ACTION]} ", style="bold")
        text.append("调用 ", style="white")
        text.append(tool, style=f"bold {self.COLORS['action']}")

        if args_display:
            text.append(f" ({args_display})", style="dim")

        if not self.compact_mode:
            self.console.print()
        self.console.print(text)

    def _render_observation(self, event: Event) -> None:
        """渲染观察结果事件"""
        tool = event.data.get("tool", "")

        # 隐藏 todo_write 工具结果，因为 TODO_UPDATE 事件已经展示了
        if tool == "todo_write":
            return

        result = str(event.data.get("result", ""))
        duration = event.data.get("duration", 0)

        if len(result) > self.max_result_chars:
            preview = result[: self.max_result_chars]
            for sep in ["\n", " ", "，", "。"]:
                last_sep = preview.rfind(sep)
                if last_sep > self.max_result_chars // 2:
                    preview = preview[:last_sep]
                    break
            remaining = len(result) - len(preview)
            display_result = f"{preview}... (+{remaining} 字符)"
        else:
            display_result = result

        header = Text()
        header.append("   └─ ", style="dim")
        header.append(tool, style=f"bold {self.COLORS['observation']}")
        header.append(f" ({duration:.1f}s)", style="dim")

        self.console.print(header)

        if "\n" in display_result:
            for line in display_result.split("\n")[:5]:
                self.console.print(f"      {line}", style="dim white")
            if display_result.count("\n") > 5:
                self.console.print("      ... (更多内容已折叠)", style="dim italic")
        else:
            self.console.print(f"      {display_result}", style="dim white")

    def _render_soft_limit(self, event: Event) -> None:
        """渲染软限制事件"""
        reason = event.data.get("reason", "")

        text = Text()
        text.append(f"{self.ICONS[EventType.SOFT_LIMIT]} ", style="bold")
        text.append("软限制触发: ", style=f"bold {self.COLORS['soft_limit']}")
        text.append(reason, style=self.COLORS["soft_limit"])

        self.console.print()
        self.console.print(text)

    def _render_error(self, event: Event) -> None:
        """渲染错误事件"""
        error = event.data.get("error", "")
        fatal = event.data.get("fatal", False)

        style = "bold red" if fatal else self.COLORS["error"]
        prefix = "致命错误" if fatal else "错误"

        panel = Panel(
            Text(error, style=style),
            title=f"{self.ICONS[EventType.ERROR]} {prefix}",
            title_align="left",
            border_style="red" if fatal else "yellow",
            padding=(0, 1),
        )
        self.console.print()
        self.console.print(panel)

    def _render_loop_end(self, event: Event) -> None:
        """渲染循环结束事件"""
        reason = event.data.get("stop_reason", "")
        content = event.data.get("content", "")

        status = Text()
        status.append(f"\n{self.ICONS[EventType.LOOP_END]} ", style="bold")
        status.append("任务完成", style=f"bold {self.COLORS['loop']}")
        status.append(f" ({reason})", style="dim")

        self.console.print(status)

        if content and len(content) > 50:
            panel = Panel(
                Markdown(content),
                title="📄 最终回答",
                title_align="left",
                border_style="green",
                padding=(1, 2),
            )
            self.console.print()
            self.console.print(panel)

    def _render_todo_update(self, event: Event) -> None:
        """渲染 Todo 更新事件"""
        todos = event.data.get("todos", [])

        if self._todos_equal(todos, self._last_todos):
            return
        self._last_todos = todos.copy() if todos else None

        if not todos:
            return

        if self.show_todo_table:
            self._render_todo_table(todos)
        else:
            self._render_todo_simple(todos)

    def _render_todo_table(self, todos: list[dict]) -> None:
        """使用表格渲染 Todo 列表"""
        table = Table(
            title="📝 任务列表",
            show_header=True,
            header_style="bold",
            border_style="dim",
            padding=(0, 1),
            collapse_padding=True,
        )

        table.add_column("#", style="dim", width=3)
        table.add_column("状态", width=4)
        table.add_column("任务", style="white")

        status_icons = {
            "pending": ("⬜", "dim"),
            "in_progress": ("🔄", "yellow"),
            "completed": ("✅", "green"),
        }

        for i, todo in enumerate(todos, 1):
            status = todo.get("status", "pending")
            icon, style = status_icons.get(status, ("❓", "white"))
            content = todo.get("content", "")

            if status == "completed":
                content_style = "dim strike"
            elif status == "in_progress":
                content_style = "bold yellow"
            else:
                content_style = "white"

            table.add_row(str(i), icon, Text(content, style=content_style))

        completed = sum(1 for t in todos if t.get("status") == "completed")
        in_progress = sum(1 for t in todos if t.get("status") == "in_progress")

        self.console.print()
        self.console.print(table)
        self.console.print(
            f"   进度: {completed}/{len(todos)} 完成, {in_progress} 进行中",
            style="dim",
        )

    def _render_todo_simple(self, todos: list[dict]) -> None:
        """简单列表形式渲染 Todo"""
        icons = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}

        tree = Tree("📝 任务列表", style="bold")

        for todo in todos:
            status = todo.get("status", "pending")
            icon = icons.get(status, "❓")
            content = todo.get("content", "")

            style = "dim" if status == "completed" else "white"
            tree.add(f"{icon} {content}", style=style)

        self.console.print()
        self.console.print(tree)

    def _render_unknown(self, event: Event) -> None:
        """渲染未知事件类型"""
        self.console.print(f"[dim][{event.type.value}] {event.data}[/dim]")

    def _format_args(self, args: dict[str, Any], max_length: int = 60) -> str:
        """格式化工具参数为简洁的显示形式"""
        if not args:
            return ""

        parts = []
        for key, value in args.items():
            if isinstance(value, str):
                if len(value) > 30:
                    value = value[:27] + "..."
                value = f'"{value}"'
            elif isinstance(value, list):
                value = f"[{len(value)} 项]"
            elif isinstance(value, dict):
                value = f"{{{len(value)} 键}}"

            parts.append(f"{key}={value}")

        result = ", ".join(parts)
        if len(result) > max_length:
            result = result[: max_length - 3] + "..."

        return result

    def _todos_equal(
        self, todos1: list[dict] | None, todos2: list[dict] | None
    ) -> bool:
        """比较两个 Todo 列表是否相同"""
        if todos1 is None and todos2 is None:
            return True
        if todos1 is None or todos2 is None:
            return False
        if len(todos1) != len(todos2):
            return False

        for t1, t2 in zip(todos1, todos2):
            if t1.get("content") != t2.get("content"):
                return False
            if t1.get("status") != t2.get("status"):
                return False

        return True


class CompactRichRenderer(RichRenderer):
    """紧凑模式的 Rich 渲染器"""

    def __init__(self, console: Console | None = None):
        super().__init__(
            console=console,
            max_thought_lines=2,
            max_result_chars=100,
            show_todo_table=False,
            compact_mode=True,
        )

    def _render_action(self, event: Event) -> None:
        """紧凑模式的工具调用渲染"""
        tool = event.data.get("tool", "")
        self.console.print(f"  → {tool}", style=f"bold {self.COLORS['action']}")

    def _render_observation(self, event: Event) -> None:
        """紧凑模式的结果渲染"""
        tool = event.data.get("tool", "")
        duration = event.data.get("duration", 0)
        self.console.print(
            f"  ← {tool} ({duration:.1f}s)", style=self.COLORS["observation"]
        )
