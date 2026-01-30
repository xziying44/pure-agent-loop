"""内置工具测试"""

import pytest
from pure_agent_loop.builtin_tools import TodoItem, TodoStore, create_todo_tool, VALID_STATUSES
from pure_agent_loop.tool import Tool


class TestTodoItem:
    """TodoItem 数据类测试"""

    def test_create_default(self):
        """默认状态应为 pending"""
        item = TodoItem(content="测试任务")
        assert item.content == "测试任务"
        assert item.status == "pending"

    def test_create_with_completed(self):
        """应支持指定 completed 状态"""
        item = TodoItem(content="已完成", status="completed")
        assert item.status == "completed"

    def test_invalid_status_raises_error(self):
        """无效状态应抛出 ValueError"""
        with pytest.raises(ValueError, match="无效的任务状态"):
            TodoItem(content="测试", status="in_progress")

    def test_invalid_status_arbitrary_string(self):
        """任意非法字符串也应抛出 ValueError"""
        with pytest.raises(ValueError, match="无效的任务状态"):
            TodoItem(content="测试", status="unknown")

    def test_to_dict(self):
        """应能转换为字典"""
        item = TodoItem(content="任务A", status="completed")
        d = item.to_dict()
        assert d == {"content": "任务A", "status": "completed"}

    def test_valid_statuses_constant(self):
        """VALID_STATUSES 应仅包含 pending 和 completed"""
        assert VALID_STATUSES == ("pending", "completed")


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
            {"content": "任务2", "status": "completed"},
        ])
        assert len(store.todos) == 2
        assert store.todos[0].content == "任务1"
        assert store.todos[1].status == "completed"

    def test_write_returns_formatted_string(self):
        """write 返回值应包含格式化的 todo 列表"""
        store = TodoStore()
        result = store.write([
            {"content": "搜索资料", "status": "completed"},
            {"content": "撰写报告", "status": "pending"},
        ])
        assert "搜索资料" in result
        assert "撰写报告" in result
        assert "✅" in result
        assert "⬜" in result

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

    def test_write_invalid_status_returns_error(self):
        """写入无效状态应返回错误提示而非抛出异常"""
        store = TodoStore()
        store.write([{"content": "旧任务", "status": "pending"}])
        result = store.write([
            {"content": "任务1", "status": "pending"},
            {"content": "任务2", "status": "in_progress"},
        ])
        assert "失败" in result
        # 写入失败，列表保持旧值
        assert len(store.todos) == 1
        assert store.todos[0].content == "旧任务"

    def test_write_invalid_status_from_empty(self):
        """从空列表写入无效状态，列表应保持为空"""
        store = TodoStore()
        result = store.write([{"content": "任务", "status": "bad"}])
        assert "失败" in result
        assert store.todos == []

    def test_format_output_summary(self):
        """格式化输出应包含正确的统计信息"""
        store = TodoStore()
        result = store.write([
            {"content": "A", "status": "pending"},
            {"content": "B", "status": "pending"},
            {"content": "C", "status": "completed"},
        ])
        assert "待处理: 2" in result
        assert "已完成: 1" in result
        assert "总计: 3" in result
        # 不应包含 "进行中" 统计
        assert "进行中" not in result


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

    def test_tool_schema_enum_values(self):
        """schema 中 status 枚举应只有 pending 和 completed"""
        store = TodoStore()
        t = create_todo_tool(store)
        schema = t.to_openai_schema()
        status_prop = schema["function"]["parameters"]["properties"]["todos"]["items"]["properties"]["status"]
        assert status_prop["enum"] == ["pending", "completed"]

    async def test_tool_execute_updates_store(self):
        """执行工具应更新 store"""
        store = TodoStore()
        t = create_todo_tool(store)
        result = await t.execute({
            "todos": [
                {"content": "任务1", "status": "pending"},
                {"content": "任务2", "status": "completed"},
            ]
        })
        assert len(store.todos) == 2
        assert "任务1" in result
        assert "任务2" in result
