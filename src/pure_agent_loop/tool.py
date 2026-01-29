"""工具系统

支持 @tool 装饰器和字典格式两种工具定义方式。
"""

import asyncio
import inspect
import re
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
    # 处理 Optional (X | None)
    origin = getattr(python_type, "__origin__", None)
    if origin is type(int | str):  # types.UnionType (Python 3.10+)
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
            if self.is_async:
                result = await self.function(**arguments)
            else:
                result = self.function(**arguments)
            return str(result)
        except Exception as e:
            return (
                f"⚠️ 工具 '{self.name}' 执行失败:\n"
                f"错误类型: {type(e).__name__}\n"
                f"错误信息: {str(e)}\n"
                f"请尝试调整参数或使用其他方法。"
            )


def tool(fn: Callable) -> Tool:
    """工具装饰器

    将普通函数转换为 Tool 对象，自动提取参数 schema 和描述。

    Args:
        fn: 要装饰的函数（同步或异步）

    Returns:
        Tool 对象
    """
    name = fn.__name__
    is_async = asyncio.iscoroutinefunction(fn)

    # 解析 docstring
    description, param_descriptions = _parse_google_docstring(fn.__doc__ or "")

    # 获取类型注解
    hints = get_type_hints(fn)
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
        name=name,
        description=description,
        parameters=parameters,
        function=fn,
        is_async=is_async,
    )


class ToolRegistry:
    """工具注册表

    管理所有注册的工具，提供查询和执行能力。
    """

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(self, tool_or_dict: Tool | dict[str, Any]) -> None:
        """注册单个工具

        支持 Tool 对象或字典格式。

        Args:
            tool_or_dict: Tool 对象或字典格式工具定义
        """
        if isinstance(tool_or_dict, Tool):
            self._tools[tool_or_dict.name] = tool_or_dict
        elif isinstance(tool_or_dict, dict):
            # 从字典格式创建 Tool
            t = Tool(
                name=tool_or_dict["name"],
                description=tool_or_dict.get("description", ""),
                parameters=tool_or_dict.get("parameters", {"type": "object", "properties": {}}),
                function=tool_or_dict["function"],
                is_async=asyncio.iscoroutinefunction(tool_or_dict["function"]),
            )
            self._tools[t.name] = t
        else:
            raise TypeError(f"不支持的工具类型: {type(tool_or_dict)}")

    def register_many(self, tools: list[Tool | dict[str, Any]]) -> None:
        """批量注册工具"""
        for t in tools:
            self.register(t)

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
