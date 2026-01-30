"""事件渲染器

将结构化事件转换为用户友好的展示格式。
支持内置默认规则和自定义渲染装饰器。
"""

from typing import Any, Callable

from .events import Event, EventType


class Renderer:
    """事件渲染器

    渲染优先级:
    1. 工具专用渲染器 (@renderer.on_tool("search"))
    2. 事件类型渲染器 (@renderer.on_event(EventType.ACTION))
    3. 内置默认渲染器
    """

    def __init__(self):
        self._tool_renderers: dict[str, Callable[[Event], str]] = {}
        self._event_renderers: dict[EventType, Callable[[Event], str]] = {}

    def on_tool(self, tool_name: str) -> Callable:
        """注册工具专用渲染器

        Args:
            tool_name: 工具名称

        Returns:
            装饰器函数
        """

        def decorator(fn: Callable[[Event], str]) -> Callable[[Event], str]:
            self._tool_renderers[tool_name] = fn
            return fn

        return decorator

    def on_event(self, event_type: EventType) -> Callable:
        """注册事件类型渲染器

        Args:
            event_type: 事件类型

        Returns:
            装饰器函数
        """

        def decorator(fn: Callable[[Event], str]) -> Callable[[Event], str]:
            self._event_renderers[event_type] = fn
            return fn

        return decorator

    def render(self, event: Event) -> str:
        """渲染事件

        按优先级匹配渲染器: 工具专用 > 事件类型 > 默认。

        Args:
            event: 要渲染的事件

        Returns:
            渲染后的字符串
        """
        # 1. 工具专用渲染器（仅 ACTION 和 OBSERVATION 事件）
        tool_name = event.data.get("tool")
        if tool_name and tool_name in self._tool_renderers:
            return self._tool_renderers[tool_name](event)

        # 2. 事件类型渲染器
        if event.type in self._event_renderers:
            return self._event_renderers[event.type](event)

        # 3. 内置默认渲染器
        return self._default_render(event)

    def _default_render(self, event: Event) -> str:
        """内置默认渲染器"""
        match event.type:
            case EventType.LOOP_START:
                return f"🚀 开始任务: {event.data.get('task', '')}"
            case EventType.THOUGHT:
                return f"💭 思考: {event.data.get('content', '')}"
            case EventType.ACTION:
                tool = event.data.get("tool", "")
                args = event.data.get("args", {})
                args_str = ", ".join(f"{k}={v!r}" for k, v in args.items())
                return f"🔧 调用工具: {tool}({args_str})"
            case EventType.OBSERVATION:
                tool = event.data.get("tool", "")
                result = event.data.get("result", "")
                duration = event.data.get("duration", 0)
                # 截断过长的结果
                preview = result[:200] + "..." if len(str(result)) > 200 else result
                return f"📋 [{tool}] 结果 ({duration:.1f}s): {preview}"
            case EventType.SOFT_LIMIT:
                return f"⚠️ 软限制触发: {event.data.get('reason', '')}"
            case EventType.ERROR:
                prefix = "❌ 致命错误" if event.data.get("fatal") else "⚠️ 错误"
                return f"{prefix}: {event.data.get('error', '')}"
            case EventType.LOOP_END:
                reason = event.data.get("stop_reason", "")
                return f"✅ 任务结束 (原因: {reason})"
            case EventType.TODO_UPDATE:
                todos = event.data.get("todos", [])
                if not todos:
                    return "📋 任务列表为空"
                icons = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}
                lines = ["📋 任务进度更新："]
                for i, t in enumerate(todos, 1):
                    icon = icons.get(t.get("status", ""), "❓")
                    lines.append(f"  {i}. {icon} {t.get('content', '')}")
                completed = sum(1 for t in todos if t.get("status") == "completed")
                lines.append(f"[{completed}/{len(todos)} 完成]")
                return "\n".join(lines)
            case _:
                return f"[{event.type.value}] {event.data}"
