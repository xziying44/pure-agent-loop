"""工具系统测试"""

import warnings

import pytest
from pure_agent_loop.tool import tool, Tool, ToolRegistry, _coerce_arguments


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


# ========== 类工具注册测试 ==========


class TestToolDecoratorDualMode:
    """@tool 装饰器双模式测试"""

    def test_method_with_self_returns_function(self):
        """类方法（有 self）应返回原始函数，不转换为 Tool"""

        class MyTools:
            @tool
            def read(self, path: str) -> str:
                """读取文件"""
                return path

        # read 应该仍是普通函数，不是 Tool
        assert not isinstance(MyTools.read, Tool)
        assert callable(MyTools.read)
        assert getattr(MyTools.read, "_tool_marker", False) is True

    def test_method_with_cls_returns_function(self):
        """类方法（有 cls）应返回原始函数"""

        class MyTools:
            @tool
            def from_config(cls, config: str) -> str:
                """从配置创建"""
                return config

        assert not isinstance(MyTools.from_config, Tool)
        assert getattr(MyTools.from_config, "_tool_marker", False) is True

    def test_standalone_function_returns_tool(self):
        """独立函数应返回 Tool（向后兼容）"""

        @tool
        def search(query: str) -> str:
            """搜索"""
            return query

        assert isinstance(search, Tool)
        assert search.name == "search"

    def test_method_still_callable_as_normal(self):
        """标记后的方法仍可正常调用"""

        class MyTools:
            @tool
            def greet(self, name: str) -> str:
                """问候"""
                return f"你好, {name}"

        instance = MyTools()
        assert instance.greet("张三") == "你好, 张三"


class TestToolClassRegistration:
    """类工具注册测试"""

    def test_register_class_instance_with_prefix(self):
        """注册类实例，使用自定义 tool_prefix"""

        class FileManager:
            tool_prefix = "fm"

            @tool
            def read_file(self, path: str) -> str:
                """读取文件

                Args:
                    path: 文件路径
                """
                return f"内容: {path}"

            @tool
            def write_file(self, path: str, content: str) -> str:
                """写入文件

                Args:
                    path: 文件路径
                    content: 写入内容
                """
                return f"已写入 {path}"

        registry = ToolRegistry()
        registry.register_class(FileManager())

        assert registry.get("fm_read_file") is not None
        assert registry.get("fm_write_file") is not None
        assert registry.get("fm_read_file").name == "fm_read_file"
        assert registry.get("fm_write_file").name == "fm_write_file"

    def test_register_class_instance_without_prefix(self):
        """注册类实例，无 tool_prefix 时使用类名"""

        class Calculator:
            @tool
            def add(self, a: int, b: int) -> str:
                """加法"""
                return str(a + b)

        registry = ToolRegistry()
        registry.register_class(Calculator())

        assert registry.get("Calculator_add") is not None

    def test_register_class_itself_auto_instantiate(self):
        """传入类本身，自动无参实例化"""

        class SimpleTools:
            tool_prefix = "st"

            @tool
            def ping(self) -> str:
                """测试连通性"""
                return "pong"

        registry = ToolRegistry()
        registry.register_class(SimpleTools)

        assert registry.get("st_ping") is not None

    def test_register_class_with_required_args_raises(self):
        """传入需要参数的类应抛出 TypeError"""

        class NeedsArgs:
            def __init__(self, config: str):
                self.config = config

            @tool
            def do_something(self, x: str) -> str:
                """做点什么"""
                return x

        with pytest.raises(TypeError, match="无法自动实例化类"):
            ToolRegistry().register_class(NeedsArgs)

    def test_register_class_no_tool_methods_warns(self):
        """类中无 @tool 方法时发出警告"""

        class EmptyTools:
            def regular_method(self):
                pass

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            ToolRegistry().register_class(EmptyTools())
            assert len(w) == 1
            assert "未发现任何 @tool 标记的方法" in str(w[0].message)

    def test_registered_tool_schema_excludes_self(self):
        """注册的工具 schema 不应包含 self 参数"""

        class MyTools:
            tool_prefix = "mt"

            @tool
            def search(self, query: str, limit: int = 10) -> str:
                """搜索内容

                Args:
                    query: 搜索关键词
                    limit: 结果数量
                """
                return ""

        registry = ToolRegistry()
        registry.register_class(MyTools())

        t = registry.get("mt_search")
        schema = t.to_openai_schema()
        params = schema["function"]["parameters"]

        # self 不应出现在参数中
        assert "self" not in params["properties"]
        # query 和 limit 应存在
        assert "query" in params["properties"]
        assert "limit" in params["properties"]
        # query 是必填，limit 不是
        assert "query" in params["required"]
        assert "limit" not in params["required"]
        # 描述应正确提取
        assert schema["function"]["description"] == "搜索内容"
        assert params["properties"]["query"].get("description") == "搜索关键词"

    @pytest.mark.asyncio
    async def test_registered_tool_executes_with_instance_state(self):
        """注册的工具执行时应能访问实例状态"""

        class Counter:
            tool_prefix = "counter"

            def __init__(self):
                self.count = 0

            @tool
            def increment(self, amount: int = 1) -> str:
                """增加计数

                Args:
                    amount: 增加量
                """
                self.count += amount
                return str(self.count)

        instance = Counter()
        registry = ToolRegistry()
        registry.register_class(instance)

        result = await registry.execute("counter_increment", {"amount": 5})
        assert result == "5"
        assert instance.count == 5

        result = await registry.execute("counter_increment", {"amount": 3})
        assert result == "8"
        assert instance.count == 8

    @pytest.mark.asyncio
    async def test_register_class_async_method(self):
        """应支持异步方法"""

        class AsyncTools:
            tool_prefix = "at"

            @tool
            async def fetch(self, url: str) -> str:
                """获取内容

                Args:
                    url: 地址
                """
                return f"响应: {url}"

        registry = ToolRegistry()
        registry.register_class(AsyncTools())

        t = registry.get("at_fetch")
        assert t.is_async is True

        result = await registry.execute("at_fetch", {"url": "https://example.com"})
        assert result == "响应: https://example.com"

    def test_register_class_skips_private_methods(self):
        """以 _ 开头的方法不会被扫描"""

        class MyTools:
            tool_prefix = "mt"

            @tool
            def public_tool(self) -> str:
                """公开工具"""
                return "public"

            @tool
            def _private_tool(self) -> str:
                """私有工具"""
                return "private"

        registry = ToolRegistry()
        registry.register_class(MyTools())

        assert registry.get("mt_public_tool") is not None
        assert registry.get("mt__private_tool") is None

    def test_register_class_with_staticmethod(self):
        """应支持 @staticmethod + @tool"""

        class Utils:
            tool_prefix = "utils"

            @staticmethod
            @tool
            def ping() -> str:
                """测试连通性"""
                return "pong"

        registry = ToolRegistry()
        registry.register_class(Utils())

        t = registry.get("utils_ping")
        assert t is not None
        assert t.name == "utils_ping"
        assert t.description == "测试连通性"

    @pytest.mark.asyncio
    async def test_staticmethod_tool_executes(self):
        """@staticmethod + @tool 注册的工具应能正确执行"""

        class Utils:
            tool_prefix = "u"

            @staticmethod
            @tool
            def add(a: int, b: int) -> str:
                """加法

                Args:
                    a: 第一个数
                    b: 第二个数
                """
                return str(a + b)

        registry = ToolRegistry()
        registry.register_class(Utils())

        result = await registry.execute("u_add", {"a": 3, "b": 7})
        assert result == "10"

    def test_register_class_with_classmethod(self):
        """应支持 @classmethod + @tool"""

        class Factory:
            tool_prefix = "factory"

            @classmethod
            @tool
            def create(cls, name: str) -> str:
                """创建对象

                Args:
                    name: 名称
                """
                return f"创建: {name} (来自 {cls.__name__})"

        registry = ToolRegistry()
        registry.register_class(Factory())

        t = registry.get("factory_create")
        assert t is not None
        # schema 中不应包含 cls
        assert "cls" not in t.parameters["properties"]
        assert "name" in t.parameters["properties"]

    @pytest.mark.asyncio
    async def test_classmethod_tool_executes(self):
        """@classmethod + @tool 注册的工具应能正确执行"""

        class Factory:
            tool_prefix = "f"

            @classmethod
            @tool
            def build(cls, item: str) -> str:
                """构建

                Args:
                    item: 项目
                """
                return f"{cls.__name__}:{item}"

        registry = ToolRegistry()
        registry.register_class(Factory())

        result = await registry.execute("f_build", {"item": "widget"})
        assert result == "Factory:widget"


class TestRegistryAutoRouting:
    """register/register_many 自动路由测试"""

    def test_register_auto_routes_class_instance(self):
        """register() 传入类实例应自动路由到 register_class"""

        class MyTools:
            tool_prefix = "my"

            @tool
            def do_it(self, x: str) -> str:
                """做事"""
                return x

        registry = ToolRegistry()
        registry.register(MyTools())

        assert registry.get("my_do_it") is not None

    def test_register_auto_routes_class_itself(self):
        """register() 传入类应自动路由到 register_class"""

        class SimpleTools:
            tool_prefix = "s"

            @tool
            def hello(self) -> str:
                """问候"""
                return "你好"

        registry = ToolRegistry()
        registry.register(SimpleTools)

        assert registry.get("s_hello") is not None

    def test_register_many_mixed_tools(self):
        """register_many 应支持混合注册"""

        @tool
        def standalone(query: str) -> str:
            """独立工具"""
            return query

        class GroupedTools:
            tool_prefix = "grp"

            @tool
            def action(self, x: str) -> str:
                """分组工具"""
                return x

        registry = ToolRegistry()
        registry.register_many([standalone, GroupedTools()])

        assert registry.get("standalone") is not None
        assert registry.get("grp_action") is not None

    def test_register_invalid_type_raises(self):
        """注册不支持的类型应抛出 TypeError"""
        registry = ToolRegistry()
        with pytest.raises(TypeError, match="不支持的工具类型"):
            registry.register("not_a_tool")

    def test_register_many_with_class_and_dict(self):
        """register_many 支持类实例和字典混合"""

        def fn(x: str) -> str:
            return x

        class Tools:
            tool_prefix = "t"

            @tool
            def op(self, v: str) -> str:
                """操作"""
                return v

        registry = ToolRegistry()
        registry.register_many([
            Tools(),
            {"name": "dict_tool", "description": "字典工具", "function": fn},
        ])

        assert registry.get("t_op") is not None
        assert registry.get("dict_tool") is not None

    def test_openai_schema_for_class_registered_tool(self):
        """类注册工具的 OpenAI schema 应格式正确"""

        class API:
            tool_prefix = "api"

            @tool
            def call(self, endpoint: str, method: str = "GET") -> str:
                """调用 API

                Args:
                    endpoint: 端点地址
                    method: HTTP 方法
                """
                return ""

        registry = ToolRegistry()
        registry.register(API())

        schemas = registry.to_openai_schemas()
        api_schema = [s for s in schemas if s["function"]["name"] == "api_call"]
        assert len(api_schema) == 1

        func = api_schema[0]["function"]
        assert func["name"] == "api_call"
        assert func["description"] == "调用 API"
        assert "endpoint" in func["parameters"]["properties"]
        assert "method" in func["parameters"]["properties"]
        assert "self" not in func["parameters"]["properties"]
        assert "endpoint" in func["parameters"]["required"]
        assert "method" not in func["parameters"]["required"]


class TestArgumentTypeCoercion:
    """参数类型强转测试"""

    def test_str_to_int(self):
        """字符串应被转换为整数"""
        props = {"offset": {"type": "integer"}}
        result = _coerce_arguments({"offset": "265"}, props)
        assert result["offset"] == 265
        assert isinstance(result["offset"], int)

    def test_str_to_float(self):
        """字符串应被转换为浮点数"""
        props = {"value": {"type": "number"}}
        result = _coerce_arguments({"value": "3.14"}, props)
        assert result["value"] == 3.14
        assert isinstance(result["value"], float)

    def test_str_to_bool(self):
        """字符串应被转换为布尔值"""
        props = {"verbose": {"type": "boolean"}}
        assert _coerce_arguments({"verbose": "true"}, props)["verbose"] is True
        assert _coerce_arguments({"verbose": "1"}, props)["verbose"] is True
        assert _coerce_arguments({"verbose": "yes"}, props)["verbose"] is True
        assert _coerce_arguments({"verbose": "false"}, props)["verbose"] is False
        assert _coerce_arguments({"verbose": "no"}, props)["verbose"] is False

    def test_str_to_array(self):
        """字符串应被转换为数组"""
        props = {"items": {"type": "array"}}
        result = _coerce_arguments({"items": '[1, 2, 3]'}, props)
        assert result["items"] == [1, 2, 3]

    def test_str_to_object(self):
        """字符串应被转换为对象"""
        props = {"config": {"type": "object"}}
        result = _coerce_arguments({"config": '{"key": "val"}'}, props)
        assert result["config"] == {"key": "val"}

    def test_already_correct_type_unchanged(self):
        """已是正确类型的值不应被重复转换"""
        props = {"offset": {"type": "integer"}, "value": {"type": "number"}}
        result = _coerce_arguments({"offset": 265, "value": 3.14}, props)
        assert result["offset"] == 265
        assert result["value"] == 3.14

    def test_conversion_failure_keeps_original(self):
        """转换失败应保持原值"""
        props = {"offset": {"type": "integer"}}
        result = _coerce_arguments({"offset": "not_a_number"}, props)
        assert result["offset"] == "not_a_number"

    def test_none_value_skipped(self):
        """None 值不应被转换"""
        props = {"offset": {"type": "integer"}}
        result = _coerce_arguments({"offset": None}, props)
        assert result["offset"] is None

    def test_unknown_param_passthrough(self):
        """不在 properties 中的参数原样保留"""
        result = _coerce_arguments({"extra": "123"}, {})
        assert result["extra"] == "123"

    @pytest.mark.asyncio
    async def test_execute_coerces_string_to_int(self):
        """端到端：Tool.execute() 应自动将字符串参数强转为正确类型"""

        @tool
        def read_file(path: str, offset: int = 0) -> str:
            """读取文件

            Args:
                path: 文件路径
                offset: 偏移量
            """
            assert isinstance(offset, int)
            return f"{path}:{offset}"

        # LLM 返回了字符串形式的 offset
        result = await read_file.execute({"path": "test.txt", "offset": "265"})
        assert result == "test.txt:265"


class TestCoerceArgumentsEnhanced:
    """参数转换增强测试"""

    def test_boolean_conversion_case_insensitive(self):
        """布尔值转换应该大小写不敏感"""
        properties = {"flag": {"type": "boolean"}}

        # 真值测试
        assert _coerce_arguments({"flag": "True"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "TRUE"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "true"}, properties)["flag"] is True

    def test_boolean_conversion_extended_formats(self):
        """布尔值转换应该支持更多格式"""
        properties = {"flag": {"type": "boolean"}}

        # 真值：yes, on, t, y
        assert _coerce_arguments({"flag": "yes"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "YES"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "on"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "ON"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "t"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "y"}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "1"}, properties)["flag"] is True

        # 假值：其他所有字符串
        assert _coerce_arguments({"flag": "false"}, properties)["flag"] is False
        assert _coerce_arguments({"flag": "no"}, properties)["flag"] is False
        assert _coerce_arguments({"flag": "off"}, properties)["flag"] is False
        assert _coerce_arguments({"flag": "0"}, properties)["flag"] is False
        assert _coerce_arguments({"flag": "random"}, properties)["flag"] is False

    def test_boolean_conversion_strips_whitespace(self):
        """布尔值转换应该去除首尾空格"""
        properties = {"flag": {"type": "boolean"}}

        assert _coerce_arguments({"flag": " true "}, properties)["flag"] is True
        assert _coerce_arguments({"flag": "  yes  "}, properties)["flag"] is True
        assert _coerce_arguments({"flag": " false "}, properties)["flag"] is False

    def test_integer_conversion_with_separators(self):
        """整数转换应该支持千分位分隔符和下划线"""
        properties = {"count": {"type": "integer"}}

        # 千分位分隔符
        assert _coerce_arguments({"count": "1,000"}, properties)["count"] == 1000
        assert _coerce_arguments({"count": "1,234,567"}, properties)["count"] == 1234567

        # Python 风格下划线
        assert _coerce_arguments({"count": "1_000"}, properties)["count"] == 1000
        assert _coerce_arguments({"count": "1_234_567"}, properties)["count"] == 1234567

        # 混合使用（虽然不推荐，但应该能处理）
        assert _coerce_arguments({"count": "1,000_000"}, properties)["count"] == 1000000

    def test_integer_conversion_strips_whitespace(self):
        """整数转换应该去除首尾空格"""
        properties = {"count": {"type": "integer"}}

        assert _coerce_arguments({"count": " 123 "}, properties)["count"] == 123
        assert _coerce_arguments({"count": "  456  "}, properties)["count"] == 456

    def test_float_conversion_with_separators(self):
        """浮点数转换应该支持千分位分隔符"""
        properties = {"price": {"type": "number"}}

        # 千分位分隔符
        assert _coerce_arguments({"price": "1,234.56"}, properties)["price"] == 1234.56
        assert _coerce_arguments({"price": "1_234.56"}, properties)["price"] == 1234.56

        # 科学计数法（原生支持）
        assert _coerce_arguments({"price": "1.23e5"}, properties)["price"] == 123000.0
        assert _coerce_arguments({"price": "1.23E5"}, properties)["price"] == 123000.0

    def test_float_conversion_strips_whitespace(self):
        """浮点数转换应该去除首尾空格"""
        properties = {"price": {"type": "number"}}

        assert _coerce_arguments({"price": " 1.23 "}, properties)["price"] == 1.23

    def test_backward_compatibility_correct_types(self):
        """已经是正确类型的参数不应该被转换"""
        properties = {
            "count": {"type": "integer"},
            "flag": {"type": "boolean"},
            "price": {"type": "number"},
        }

        args = {"count": 123, "flag": True, "price": 1.23}
        result = _coerce_arguments(args, properties)

        assert result["count"] == 123
        assert result["flag"] is True
        assert result["price"] == 1.23

    def test_backward_compatibility_none_values(self):
        """None 值应该保持 None"""
        properties = {"count": {"type": "integer"}}

        result = _coerce_arguments({"count": None}, properties)
        assert result["count"] is None

    def test_backward_compatibility_conversion_failure(self):
        """转换失败时应该保留原值"""
        properties = {"count": {"type": "integer"}}

        # 无法转换的字符串应该保留原值
        result = _coerce_arguments({"count": "abc"}, properties)
        assert result["count"] == "abc"

    def test_backward_compatibility_unknown_parameters(self):
        """不在 schema 中的参数应该原样保留"""
        properties = {"count": {"type": "integer"}}

        result = _coerce_arguments({"count": "123", "unknown": "value"}, properties)
        assert result["count"] == 123
        assert result["unknown"] == "value"
