"""渲染器测试"""

import pytest
from pure_agent_loop.renderer import Renderer
from pure_agent_loop.events import Event, EventType


class TestRenderer:
    """Renderer 测试"""

    def test_default_render_action(self):
        """默认应渲染 action 事件"""
        renderer = Renderer()
        event = Event.action(step=1, tool="search", args={"query": "python"})
        result = renderer.render(event)
        assert "search" in result
        assert result is not None

    def test_default_render_observation(self):
        """默认应渲染 observation 事件"""
        renderer = Renderer()
        event = Event.observation(step=1, tool="search", result="找到结果", duration=1.2)
        result = renderer.render(event)
        assert result is not None

    def test_default_render_thought(self):
        """默认应渲染 thought 事件"""
        renderer = Renderer()
        event = Event.thought(step=1, content="让我思考一下")
        result = renderer.render(event)
        assert "让我思考一下" in result

    def test_default_render_error(self):
        """默认应渲染 error 事件"""
        renderer = Renderer()
        event = Event.error(step=1, error="出错了")
        result = renderer.render(event)
        assert "出错了" in result

    def test_custom_tool_renderer(self):
        """应该支持自定义工具渲染器"""
        renderer = Renderer()

        @renderer.on_tool("search")
        def render_search(event: Event) -> str:
            return f"🔍 搜索: {event.data['args']['query']}"

        event = Event.action(step=1, tool="search", args={"query": "python"})
        result = renderer.render(event)
        assert result == "🔍 搜索: python"

    def test_custom_event_renderer(self):
        """应该支持自定义事件类型渲染器"""
        renderer = Renderer()

        @renderer.on_event(EventType.SOFT_LIMIT)
        def render_limit(event: Event) -> str:
            return f"⚠️ 限制: {event.data['reason']}"

        event = Event.soft_limit(step=10, reason="step_limit", prompt="请调整")
        result = renderer.render(event)
        assert result == "⚠️ 限制: step_limit"

    def test_tool_renderer_priority(self):
        """工具专用渲染器应优先于事件类型渲染器"""
        renderer = Renderer()

        @renderer.on_tool("search")
        def render_search(event: Event) -> str:
            return "工具专用"

        @renderer.on_event(EventType.ACTION)
        def render_action(event: Event) -> str:
            return "类型通用"

        event = Event.action(step=1, tool="search", args={})
        result = renderer.render(event)
        assert result == "工具专用"

    def test_event_renderer_fallback(self):
        """无工具专用渲染器时应回退到事件类型渲染器"""
        renderer = Renderer()

        @renderer.on_event(EventType.ACTION)
        def render_action(event: Event) -> str:
            return "类型通用"

        event = Event.action(step=1, tool="unknown_tool", args={})
        result = renderer.render(event)
        assert result == "类型通用"

    def test_render_returns_none_for_unhandled(self):
        """无渲染器的事件类型返回默认渲染"""
        renderer = Renderer()
        event = Event.loop_start(task="测试")
        result = renderer.render(event)
        # 默认渲染器应该返回某些内容
        assert isinstance(result, str)
