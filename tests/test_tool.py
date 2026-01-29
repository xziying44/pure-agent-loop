"""工具系统测试"""

import pytest
from pure_agent_loop.tool import tool, Tool, ToolRegistry


class TestToolDecorator:
    """@tool 装饰器测试"""

    def test_decorate_sync_function(self):
        """应该能装饰同步函数"""

        @tool
        def search(query: str) -> str:
            """搜索网页内容"""
            return f"结果: {query}"

        assert isinstance(search, Tool)
        assert search.name == "search"
        assert search.description == "搜索网页内容"

    def test_decorate_async_function(self):
        """应该能装饰异步函数"""

        @tool
        async def search(query: str) -> str:
            """搜索网页内容"""
            return f"结果: {query}"

        assert isinstance(search, Tool)
        assert search.name == "search"

    def test_extract_parameters_schema(self):
        """应该从类型注解提取参数 schema"""

        @tool
        def search(query: str, max_results: int = 5) -> str:
            """搜索网页内容

            Args:
                query: 搜索关键词
                max_results: 最大返回结果数
            """
            return ""

        schema = search.to_openai_schema()
        params = schema["function"]["parameters"]
        assert params["type"] == "object"
        assert "query" in params["properties"]
        assert params["properties"]["query"]["type"] == "string"
        assert params["properties"]["max_results"]["type"] == "integer"
        assert "query" in params["required"]
        assert "max_results" not in params["required"]

    def test_extract_description_from_docstring(self):
        """应该从 docstring 提取描述"""

        @tool
        def search(query: str) -> str:
            """搜索网页内容

            Args:
                query: 搜索关键词
            """
            return ""

        schema = search.to_openai_schema()
        assert schema["function"]["description"] == "搜索网页内容"
        assert schema["function"]["parameters"]["properties"]["query"].get("description") == "搜索关键词"

    def test_optional_parameter(self):
        """应该处理 Optional 参数"""

        @tool
        def search(query: str, lang: str | None = None) -> str:
            """搜索"""
            return ""

        schema = search.to_openai_schema()
        assert "lang" in schema["function"]["parameters"]["properties"]
        assert "lang" not in schema["function"]["parameters"]["required"]

    def test_bool_parameter(self):
        """应该处理布尔参数"""

        @tool
        def search(query: str, verbose: bool = False) -> str:
            """搜索"""
            return ""

        schema = search.to_openai_schema()
        assert schema["function"]["parameters"]["properties"]["verbose"]["type"] == "boolean"

    def test_float_parameter(self):
        """应该处理浮点参数"""

        @tool
        def calc(value: float) -> str:
            """计算"""
            return ""

        schema = calc.to_openai_schema()
        assert schema["function"]["parameters"]["properties"]["value"]["type"] == "number"


class TestTool:
    """Tool 对象测试"""

    @pytest.mark.asyncio
    async def test_execute_sync_tool(self):
        """应该能执行同步工具"""

        @tool
        def add(a: int, b: int) -> str:
            """加法"""
            return str(a + b)

        result = await add.execute({"a": 1, "b": 2})
        assert result == "3"

    @pytest.mark.asyncio
    async def test_execute_async_tool(self):
        """应该能执行异步工具"""

        @tool
        async def add(a: int, b: int) -> str:
            """加法"""
            return str(a + b)

        result = await add.execute({"a": 1, "b": 2})
        assert result == "3"

    @pytest.mark.asyncio
    async def test_execute_error_returns_message(self):
        """工具执行失败应返回错误信息字符串"""

        @tool
        def bad_tool(x: str) -> str:
            """坏工具"""
            raise ValueError("参数无效")

        result = await bad_tool.execute({"x": "test"})
        assert "执行失败" in result
        assert "ValueError" in result
        assert "参数无效" in result


class TestToolRegistry:
    """ToolRegistry 测试"""

    def test_register_tool_decorator(self):
        """应该能注册 @tool 装饰的函数"""

        @tool
        def search(query: str) -> str:
            """搜索"""
            return ""

        registry = ToolRegistry()
        registry.register(search)
        assert registry.get("search") is search

    def test_register_dict_format(self):
        """应该能注册字典格式的工具"""

        def search_fn(query: str) -> str:
            return ""

        tool_dict = {
            "name": "search",
            "description": "搜索网页",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "关键词"}
                },
                "required": ["query"],
            },
            "function": search_fn,
        }

        registry = ToolRegistry()
        registry.register(tool_dict)
        assert registry.get("search") is not None
        assert registry.get("search").name == "search"

    def test_register_many(self):
        """应该能批量注册"""

        @tool
        def tool_a(x: str) -> str:
            """工具A"""
            return ""

        @tool
        def tool_b(x: str) -> str:
            """工具B"""
            return ""

        registry = ToolRegistry()
        registry.register_many([tool_a, tool_b])
        assert registry.get("tool_a") is not None
        assert registry.get("tool_b") is not None

    def test_to_openai_schemas(self):
        """应该能转换为 OpenAI tools 格式"""

        @tool
        def search(query: str) -> str:
            """搜索"""
            return ""

        registry = ToolRegistry()
        registry.register(search)
        schemas = registry.to_openai_schemas()
        assert len(schemas) == 1
        assert schemas[0]["type"] == "function"
        assert schemas[0]["function"]["name"] == "search"

    @pytest.mark.asyncio
    async def test_execute_tool(self):
        """应该能按名称执行工具"""

        @tool
        def add(a: int, b: int) -> str:
            """加法"""
            return str(a + b)

        registry = ToolRegistry()
        registry.register(add)
        result = await registry.execute("add", {"a": 3, "b": 4})
        assert result == "7"

    def test_get_unknown_tool_returns_none(self):
        """获取不存在的工具应返回 None"""
        registry = ToolRegistry()
        assert registry.get("unknown") is None
