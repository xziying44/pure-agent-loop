"""内置工具测试"""

import pytest
from pure_agent_loop.builtin_tools import (
    TodoItem, TodoStore, create_todo_tool, VALID_STATUSES, VALID_PRIORITIES,
)
from pure_agent_loop.tool import Tool


class TestTodoItem:
    """TodoItem 数据类测试"""

    def test_create_default(self):
        """默认状态为 pending、优先级为 medium"""
        item = TodoItem(content="测试任务")
        assert item.status == "pending"
        assert item.priority == "medium"

    def test_all_valid_statuses(self):
        """四种状态都应合法"""
        for status in ("pending", "in_progress", "completed", "cancelled"):
            item = TodoItem(content="任务", status=status)
            assert item.status == status

    def test_all_valid_priorities(self):
        """三种优先级都应合法"""
        for priority in ("high", "medium", "low"):
            item = TodoItem(content="任务", priority=priority)
            assert item.priority == priority

    def test_invalid_status_raises_error(self):
        """无效状态应抛出 ValueError"""
        with pytest.raises(ValueError, match="无效的任务状态"):
            TodoItem(content="测试", status="unknown")

    def test_invalid_priority_raises_error(self):
        """无效优先级应抛出 ValueError"""
        with pytest.raises(ValueError, match="无效的优先级"):
            TodoItem(content="测试", priority="urgent")

    def test_to_dict_includes_priority(self):
        """to_dict 应包含 priority 字段"""
        item = TodoItem(content="A", status="in_progress", priority="high")
        d = item.to_dict()
        assert d == {"content": "A", "status": "in_progress", "priority": "high"}

    def test_valid_statuses_constant(self):
        """VALID_STATUSES 应包含四种状态"""
        assert VALID_STATUSES == ("pending", "in_progress", "completed", "cancelled")

    def test_valid_priorities_constant(self):
        """VALID_PRIORITIES 应包含三种优先级"""
        assert VALID_PRIORITIES == ("high", "medium", "low")


class TestTodoStore:
    """TodoStore 测试"""

    def test_initial_empty(self):
        store = TodoStore()
        assert store.todos == []

    def test_write_with_in_progress(self):
        """支持 in_progress 状态"""
        store = TodoStore()
        store.write([
            {"content": "正在做", "status": "in_progress"},
        ])
        assert store.todos[0].status == "in_progress"

    def test_write_with_cancelled(self):
        """支持 cancelled 状态"""
        store = TodoStore()
        store.write([{"content": "已取消", "status": "cancelled"}])
        assert store.todos[0].status == "cancelled"

    def test_write_with_priority(self):
        """支持 priority 字段"""
        store = TodoStore()
        store.write([{"content": "紧急任务", "status": "pending", "priority": "high"}])
        assert store.todos[0].priority == "high"

    def test_write_default_priority(self):
        """未指定 priority 时默认 medium"""
        store = TodoStore()
        store.write([{"content": "任务", "status": "pending"}])
        assert store.todos[0].priority == "medium"

    def test_write_replaces_list(self):
        store = TodoStore()
        store.write([
            {"content": "A", "status": "pending"},
            {"content": "B", "status": "completed"},
        ])
        assert len(store.todos) == 2

    def test_write_returns_formatted_string(self):
        store = TodoStore()
        result = store.write([
            {"content": "搜索", "status": "completed"},
            {"content": "分析", "status": "in_progress"},
            {"content": "报告", "status": "pending"},
        ])
        assert "✅" in result
        assert "🔵" in result
        assert "⬜" in result

    def test_write_empty_list(self):
        store = TodoStore()
        store.write([{"content": "A", "status": "pending"}])
        result = store.write([])
        assert store.todos == []
        assert "空" in result

    def test_write_invalid_status_returns_error(self):
        """写入无效状态应返回错误，保持旧列表"""
        store = TodoStore()
        store.write([{"content": "旧任务", "status": "pending"}])
        result = store.write([{"content": "新任务", "status": "bad"}])
        assert "失败" in result
        assert len(store.todos) == 1
        assert store.todos[0].content == "旧任务"

    def test_todos_property_returns_copy(self):
        store = TodoStore()
        store.write([{"content": "A", "status": "pending"}])
        store.todos.clear()
        assert len(store.todos) == 1

    def test_format_output_summary_four_statuses(self):
        """格式化输出应显示四种状态的统计"""
        store = TodoStore()
        result = store.write([
            {"content": "A", "status": "pending"},
            {"content": "B", "status": "in_progress"},
            {"content": "C", "status": "completed"},
            {"content": "D", "status": "cancelled"},
        ])
        assert "待处理: 1" in result
        assert "进行中: 1" in result
        assert "已完成: 1" in result
        assert "已取消: 1" in result


class TestCreateTodoTool:
    """create_todo_tool 工厂函数测试"""

    def test_creates_tool_instance(self):
        store = TodoStore()
        t = create_todo_tool(store)
        assert isinstance(t, Tool)
        assert t.name == "todo_write"

    def test_tool_schema_has_four_statuses(self):
        """schema 中 status 应有四种枚举值"""
        store = TodoStore()
        t = create_todo_tool(store)
        schema = t.to_openai_schema()
        status_prop = schema["function"]["parameters"]["properties"]["todos"]["items"]["properties"]["status"]
        assert set(status_prop["enum"]) == {"pending", "in_progress", "completed", "cancelled"}

    def test_tool_schema_has_priority(self):
        """schema 中应包含 priority 字段"""
        store = TodoStore()
        t = create_todo_tool(store)
        schema = t.to_openai_schema()
        item_props = schema["function"]["parameters"]["properties"]["todos"]["items"]["properties"]
        assert "priority" in item_props

    async def test_tool_execute_updates_store(self):
        store = TodoStore()
        t = create_todo_tool(store)
        await t.execute({
            "todos": [
                {"content": "A", "status": "in_progress", "priority": "high"},
                {"content": "B", "status": "pending"},
            ]
        })
        assert len(store.todos) == 2
        assert store.todos[0].priority == "high"
        assert store.todos[1].priority == "medium"  # 默认值

    def test_tool_description_length(self):
        """工具描述应足够详细（参考 opencode 的 167 行描述）"""
        store = TodoStore()
        t = create_todo_tool(store)
        assert len(t.description) > 200
