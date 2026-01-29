"""错误类型测试"""

import pytest
from pure_agent_loop.errors import (
    PureAgentLoopError,
    ToolExecutionError,
    LLMError,
    LimitExceededError,
)


class TestErrors:
    """异常类测试"""

    def test_base_error_is_exception(self):
        """基础异常应该继承自 Exception"""
        error = PureAgentLoopError("test error")
        assert isinstance(error, Exception)
        assert str(error) == "test error"

    def test_tool_execution_error(self):
        """工具执行异常应该包含工具名称"""
        error = ToolExecutionError("search", "connection timeout")
        assert error.tool_name == "search"
        assert "search" in str(error)
        assert "connection timeout" in str(error)

    def test_llm_error(self):
        """LLM 异常应该包含原始异常"""
        original = ValueError("rate limit")
        error = LLMError("API 调用失败", original)
        assert error.original_error == original
        assert "API 调用失败" in str(error)

    def test_limit_exceeded_error(self):
        """限制超出异常应该包含限制类型"""
        error = LimitExceededError("token_limit", 100000, 150000)
        assert error.limit_type == "token_limit"
        assert error.limit_value == 100000
        assert error.current_value == 150000
