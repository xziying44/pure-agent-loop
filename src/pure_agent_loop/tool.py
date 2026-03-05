"""工具系统

支持 @tool 装饰器和字典格式两种工具定义方式。
"""

import asyncio
import inspect
import json
import re
import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, get_type_hints


# Python 类型到 JSON Schema 类型的映射
_TYPE_MAP: dict[type, str] = {
    str: "string",
    int: "integer",
    float: "number",
    bool: "boolean",
    list: "array",
    dict: "object",
}


def _coerce_arguments(
    arguments: dict[str, Any], properties: dict[str, Any]
) -> dict[str, Any]:
    """根据 JSON Schema 类型定义，对字符串值做宽容强转

    仅对 isinstance(value, str) 的值尝试转换，已是正确类型或不在 schema 中的参数原样保留。
    转换失败时保持原值，让函数自己报错。

    Args:
        arguments: 工具参数字典
        properties: JSON Schema 的 properties 定义

    Returns:
        强转后的参数字典
    """
    coerced = {}
    for key, value in arguments.items():
        # None 值、非字符串值、不在 schema 中的参数，原样保留
        if value is None or not isinstance(value, str) or key not in properties:
            coerced[key] = value
            continue

        expected_type = properties[key].get("type")
        try:
            if expected_type == "integer":
                cleaned = value.replace(",", "").replace("_", "").strip()
                coerced[key] = int(cleaned)
            elif expected_type == "number":
                cleaned = value.replace(",", "").replace("_", "").strip()
                coerced[key] = float(cleaned)
            elif expected_type == "boolean":
                normalized = value.lower().strip()
                coerced[key] = normalized in ("true", "1", "yes", "on", "t", "y")
            elif expected_type in ("array", "object"):
                coerced[key] = json.loads(value)
            else:
                coerced[key] = value
        except (ValueError, json.JSONDecodeError):
            # 转换失败保持原值
            coerced[key] = value
    return coerced


def _parse_google_docstring(docstring: str) -> tuple[str, dict[str, str]]:
    """解析 Google 风格的 docstring

    Returns:
        (描述, {参数名: 参数描述})
    """
    if not docstring:
        return "", {}

    lines = docstring.strip().split("\n")
    description_lines = []
    param_descriptions: dict[str, str] = {}
    in_args_section = False

    for line in lines:
        stripped = line.strip()

        # 检测 Args: 段落
        if stripped.lower().startswith("args:"):
            in_args_section = True
            continue

        if in_args_section:
            # 新段落开始（如 Returns:, Raises:）则结束 Args 解析
            if stripped and not stripped.startswith("-") and ":" in stripped:
                # 检查是否是参数行（格式: param_name: description）
                match = re.match(r"(\w+)\s*(?:\(.*?\))?\s*:\s*(.+)", stripped)
                if match:
                    param_descriptions[match.group(1)] = match.group(2).strip()
                    continue
            if stripped.lower() in ("returns:", "raises:", "yields:", "examples:", "note:", "notes:"):
                in_args_section = False
                continue
            # 可能是参数描述的续行，忽略
            continue

        if stripped:
            description_lines.append(stripped)

    description = " ".join(description_lines) if description_lines else ""
    return description, param_descriptions


def _get_json_type(python_type: type) -> str:
    """将 Python 类型转换为 JSON Schema 类型"""
    # 处理 Optional (X | None) - Python 3.10+ 的 UnionType
    if isinstance(python_type, type(int | str)):  # types.UnionType
        args = python_type.__args__
        non_none = [a for a in args if a is not type(None)]
        if non_none:
            return _get_json_type(non_none[0])

    return _TYPE_MAP.get(python_type, "string")


class Tool:
    """工具对象

    封装工具的元信息和执行逻辑。

    Attributes:
        name: 工具名称
        description: 工具描述
        parameters: JSON Schema 格式的参数描述
        function: 实际执行的函数
        is_async: 是否为异步函数
    """

    def __init__(
        self,
        name: str,
        description: str,
        parameters: dict[str, Any],
        function: Callable,
        is_async: bool = False,
    ):
        self.name = name
        self.description = description
        self.parameters = parameters
        self.function = function
        self.is_async = is_async

    def to_openai_schema(self) -> dict[str, Any]:
        """转换为 OpenAI Function Calling 格式"""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    async def execute(self, arguments: dict[str, Any]) -> str:
        """执行工具函数

        出错时不抛出异常，而是返回格式化的错误信息。

        Args:
            arguments: 工具参数字典

        Returns:
            工具执行结果（字符串）
        """
        try:
            coerced = _coerce_arguments(
                arguments, self.parameters.get("properties", {})
            )
            if self.is_async:
                result = await self.function(**coerced)
            else:
                result = self.function(**coerced)
            return str(result)
        except Exception as e:
            return (
                f"⚠️ 工具 '{self.name}' 执行失败:\n"
                f"错误类型: {type(e).__name__}\n"
                f"错误信息: {str(e)}\n"
                f"请尝试调整参数或使用其他方法。"
            )


def _build_tool(fn: Callable, *, name: str | None = None) -> Tool:
    """从函数构建 Tool 对象（内部辅助函数）

    将函数的类型注解和 docstring 自动转换为 Tool 对象。
    支持普通函数和绑定方法。

    Args:
        fn: 要转换的函数（同步或异步，支持绑定方法）
        name: 自定义工具名称，默认使用函数名

    Returns:
        Tool 对象
    """
    tool_name = name or fn.__name__
    is_async = asyncio.iscoroutinefunction(fn)

    # 解析 docstring
    description, param_descriptions = _parse_google_docstring(fn.__doc__ or "")

    # 获取类型注解（绑定方法需从 __func__ 获取）
    func = getattr(fn, "__func__", fn)
    hints = get_type_hints(func)
    sig = inspect.signature(fn)

    # 构建 JSON Schema
    properties: dict[str, Any] = {}
    required: list[str] = []

    for param_name, param in sig.parameters.items():
        if param_name == "return":
            continue

        param_type = hints.get(param_name, str)
        json_type = _get_json_type(param_type)

        prop: dict[str, Any] = {"type": json_type}

        # 添加参数描述
        if param_name in param_descriptions:
            prop["description"] = param_descriptions[param_name]

        properties[param_name] = prop

        # 无默认值 => 必填参数
        if param.default is inspect.Parameter.empty:
            required.append(param_name)

    parameters = {
        "type": "object",
        "properties": properties,
        "required": required,
    }

    return Tool(
        name=tool_name,
        description=description,
        parameters=parameters,
        function=fn,
        is_async=is_async,
    )


def tool(fn: Callable) -> Tool | Callable:
    """工具装饰器

    双模式行为：
    - 独立函数（无 self/cls）：转换为 Tool 对象
    - 类方法（有 self/cls）：仅标记 _tool_marker = True，返回原始函数

    Args:
        fn: 要装饰的函数（同步或异步）

    Returns:
        Tool 对象（独立函数）或标记后的原始函数（类方法）
    """
    sig = inspect.signature(fn)
    params = list(sig.parameters.keys())

    # 类方法模式：首个参数为 self/cls，仅标记不转换
    if params and params[0] in ("self", "cls"):
        fn._tool_marker = True  # type: ignore[attr-defined]
        return fn

    # 独立函数模式：转换为 Tool 对象
    return _build_tool(fn)


def _has_tool_methods(cls: type) -> bool:
    """检查类中是否包含 @tool 标记的方法

    Args:
        cls: 要检查的类

    Returns:
        是否包含工具方法
    """
    for name, value in cls.__dict__.items():
        if name.startswith("_"):
            continue
        # 普通方法或异步方法 + @tool 标记
        if callable(value) and getattr(value, "_tool_marker", False):
            return True
        # @staticmethod + @tool（内层 @tool 转换为 Tool 对象）
        if isinstance(value, staticmethod) and isinstance(value.__func__, Tool):
            return True
        # @classmethod + @tool（内层 @tool 标记了函数）
        if isinstance(value, classmethod) and getattr(value.__func__, "_tool_marker", False):
            return True
    return False


class ToolRegistry:
    """工具注册表

    管理所有注册的工具，提供查询和执行能力。
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, item: Any) -> None:
        """注册单个工具

        支持 Tool 对象、字典格式、类或包含 @tool 方法的类实例。

        Args:
            item: Tool 对象、字典格式工具定义、类或类实例
        """
        if isinstance(item, Tool):
            self._tools[item.name] = item
        elif isinstance(item, dict):
            # 从字典格式创建 Tool
            t = Tool(
                name=item["name"],
                description=item.get("description", ""),
                parameters=item.get("parameters", {"type": "object", "properties": {}}),
                function=item["function"],
                is_async=asyncio.iscoroutinefunction(item["function"]),
            )
            self._tools[t.name] = t
        elif isinstance(item, type):
            # 传入类本身
            self.register_class(item)
        elif not isinstance(item, (str, int, float, bool, list, tuple, set)) and _has_tool_methods(type(item)):
            # 传入包含 @tool 方法的类实例
            self.register_class(item)
        else:
            raise TypeError(f"不支持的工具类型: {type(item)}")

    def register_many(self, tools: list[Any]) -> None:
        """批量注册工具"""
        for t in tools:
            self.register(t)

    def register_class(self, obj: Any) -> None:
        """注册类中所有 @tool 标记的方法

        支持传入类本身（自动无参实例化）或类实例。
        工具名称格式：{tool_prefix 或类名}_{方法名}

        Args:
            obj: 类或类实例
        """
        # 解析类和实例
        if isinstance(obj, type):
            cls = obj
            try:
                instance = cls()
            except TypeError as e:
                raise TypeError(
                    f"无法自动实例化类 '{cls.__name__}'：构造函数需要参数。"
                    f"请传入类的实例而非类本身。"
                ) from e
        else:
            instance = obj
            cls = type(obj)

        # 获取工具名称前缀
        prefix = getattr(cls, "tool_prefix", cls.__name__)

        # 扫描并注册标记的方法
        found = False
        for attr_name, raw_value in cls.__dict__.items():
            if attr_name.startswith("_"):
                continue

            tool_name = f"{prefix}_{attr_name}"

            # 情况1：普通方法 + @tool 标记
            if callable(raw_value) and getattr(raw_value, "_tool_marker", False):
                found = True
                bound_method = getattr(instance, attr_name)
                self._tools[tool_name] = _build_tool(bound_method, name=tool_name)
                continue

            # 情况2：@staticmethod + @tool（@tool 返回了 Tool 对象）
            if isinstance(raw_value, staticmethod) and isinstance(
                raw_value.__func__, Tool
            ):
                found = True
                original_tool = raw_value.__func__
                # 复用已有 Tool 的 schema，仅重命名
                self._tools[tool_name] = Tool(
                    name=tool_name,
                    description=original_tool.description,
                    parameters=original_tool.parameters,
                    function=original_tool.function,
                    is_async=original_tool.is_async,
                )
                continue

            # 情况3：@classmethod + @tool 标记
            if isinstance(raw_value, classmethod) and getattr(
                raw_value.__func__, "_tool_marker", False
            ):
                found = True
                bound_method = getattr(instance, attr_name)
                self._tools[tool_name] = _build_tool(bound_method, name=tool_name)
                continue

        if not found:
            warnings.warn(f"类 '{cls.__name__}' 中未发现任何 @tool 标记的方法")

    def get(self, name: str) -> Tool | None:
        """按名称获取工具"""
        return self._tools.get(name)

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        """转换为 OpenAI tools 格式列表"""
        return [t.to_openai_schema() for t in self._tools.values()]

    async def execute(self, name: str, arguments: dict[str, Any]) -> str:
        """按名称执行工具

        Args:
            name: 工具名称
            arguments: 工具参数

        Returns:
            执行结果字符串
        """
        t = self.get(name)
        if t is None:
            return f"⚠️ 未知工具 '{name}'，请检查工具名称。"
        return await t.execute(arguments)
