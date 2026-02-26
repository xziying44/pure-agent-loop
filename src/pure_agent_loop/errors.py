"""自定义异常类型

定义 pure-agent-loop 框架使用的所有异常类型。
"""

from typing import Any


class PureAgentLoopError(Exception):
    """pure-agent-loop 框架的基础异常类"""

    pass


class ToolExecutionError(PureAgentLoopError):
    """工具执行失败异常

    当工具函数执行过程中发生错误时抛出。
    """

    def __init__(self, tool_name: str, message: str):
        self.tool_name = tool_name
        self.message = message
        super().__init__(f"工具 '{tool_name}' 执行失败: {message}")


class LLMError(PureAgentLoopError):
    """LLM 调用异常

    当 LLM API 调用失败且重试耗尽时抛出。
    """

    def __init__(self, message: str, original_error: Exception | None = None):
        self.original_error = original_error
        super().__init__(message)


class LimitExceededError(PureAgentLoopError):
    """限制超出异常

    当硬限制被触发时抛出。
    """

    def __init__(self, limit_type: str, limit_value: Any, current_value: Any):
        self.limit_type = limit_type
        self.limit_value = limit_value
        self.current_value = current_value
        super().__init__(
            f"超出 {limit_type} 限制: 当前 {current_value}, 上限 {limit_value}"
        )


class SandboxViolationError(PureAgentLoopError):
    """沙箱权限违规异常

    当文件操作的目标路径不在沙箱允许范围内时抛出。
    """

    def __init__(self, path: str, operation: str, allowed_paths: list[str]):
        self.path = path
        self.operation = operation
        self.allowed_paths = allowed_paths
        super().__init__(
            f"沙箱权限不足: 无法对 '{path}' 执行 {operation} 操作。"
            f"允许的路径: {allowed_paths}"
        )
