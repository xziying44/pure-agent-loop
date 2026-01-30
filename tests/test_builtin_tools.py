"""内置工具测试"""

import pytest
from pure_agent_loop.builtin_tools import TodoItem, TodoStore, create_todo_tool
from pure_agent_loop.tool import Tool


class TestTodoItem:
    """TodoItem 数据类测试"""

    def test_create_default(self):
        """默认状态应为 pending"""
        item = TodoItem(content="测试任务")
        assert item.content == "测试任务"
        assert item.status == "pending"

    def test_create_with_status(self):
        """应支持指定状态"""
        item = TodoItem(content="进行中", status="in_progress")
        assert item.status == "in_progress"

    def test_to_dict(self):
        """应能转换为字典"""
        item = TodoItem(content="任务A", status="completed")
        d = item.to_dict()
        assert d == {"content": "任务A", "status": "completed"}


class TestTodoStore:
    """TodoStore 测试"""

    def test_initial_empty(self):
        """初始应为空列表"""
        store = TodoStore()
        assert store.todos == []

    def test_write_replaces_list(self):
        """write 应完全替换 todo 列表"""
        store = TodoStore()
        store.write([
            {"content": "任务1", "status": "pending"},
            {"content": "任务2", "status": "in_progress"},
        ])
        assert len(store.todos) == 2
        assert store.todos[0].content == "任务1"
        assert store.todos[1].status == "in_progress"

    def test_write_returns_formatted_string(self):
        """write 返回值应包含格式化的 todo 列表"""
        store = TodoStore()
        result = store.write([
            {"content": "搜索资料", "status": "completed"},
            {"content": "分析结果", "status": "in_progress"},
            {"content": "撰写报告", "status": "pending"},
        ])
        assert "搜索资料" in result
        assert "分析结果" in result
        assert "撰写报告" in result
        assert "completed" in result or "✅" in result

    def test_write_empty_list(self):
        """写入空列表应清空"""
        store = TodoStore()
        store.write([{"content": "任务", "status": "pending"}])
        result = store.write([])
        assert store.todos == []
        assert "空" in result

    def test_multiple_writes_replace(self):
        """多次 write 应完全替换"""
        store = TodoStore()
        store.write([{"content": "A", "status": "pending"}])
        store.write([{"content": "B", "status": "completed"}])
        assert len(store.todos) == 1
        assert store.todos[0].content == "B"

    def test_todos_property_returns_copy(self):
        """todos 属性应返回副本，不影响内部状态"""
        store = TodoStore()
        store.write([{"content": "A", "status": "pending"}])
        external = store.todos
        external.clear()
        assert len(store.todos) == 1


class TestCreateTodoTool:
    """create_todo_tool 工厂函数测试"""

    def test_creates_tool_instance(self):
        """应返回 Tool 实例"""
        store = TodoStore()
        t = create_todo_tool(store)
        assert isinstance(t, Tool)
        assert t.name == "todo_write"

    def test_tool_has_correct_schema(self):
        """工具 schema 应正确定义"""
        store = TodoStore()
        t = create_todo_tool(store)
        schema = t.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "todo_write"
        params = schema["function"]["parameters"]
        assert "todos" in params["properties"]

    async def test_tool_execute_updates_store(self):
        """执行工具应更新 store"""
        store = TodoStore()
        t = create_todo_tool(store)
        result = await t.execute({
            "todos": [
                {"content": "任务1", "status": "pending"},
                {"content": "任务2", "status": "in_progress"},
            ]
        })
        assert len(store.todos) == 2
        assert "任务1" in result
        assert "任务2" in result
