"""重试机制

为 LLM API 调用提供自动重试与指数退避能力。
"""

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Awaitable


# 默认可重试的异常类型
DEFAULT_RETRYABLE_ERRORS = (
    ConnectionError,
    TimeoutError,
    OSError,
)


@dataclass
class RetryConfig:
    """重试配置

    Attributes:
        max_retries: 最大重试次数（默认 3）
        base_delay: 基础延迟秒数（默认 1.0）
        max_delay: 最大延迟秒数（默认 30.0）
        retryable_errors: 可重试的异常类型元组
    """

    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    retryable_errors: tuple[type[Exception], ...] = DEFAULT_RETRYABLE_ERRORS


class RetryHandler:
    """重试处理器

    使用指数退避策略自动重试失败的异步调用。

    Args:
        config: 重试配置
    """

    def __init__(self, config: RetryConfig | None = None):
        self.config = config or RetryConfig()

    def _calculate_delay(self, attempt: int) -> float:
        """计算退避延迟时间

        Args:
            attempt: 当前重试次数（0-based）

        Returns:
            延迟秒数
        """
        delay = self.config.base_delay * (2 ** attempt)
        return min(delay, self.config.max_delay)

    def _is_retryable(self, error: Exception) -> bool:
        """判断异常是否可重试"""
        return isinstance(error, self.config.retryable_errors)

    async def execute(
        self,
        fn: Callable[..., Awaitable[Any]],
        *args: Any,
        on_retry: Callable[[int, Exception, float], None] | None = None,
        **kwargs: Any,
    ) -> Any:
        """执行异步函数，失败时自动重试

        Args:
            fn: 要执行的异步函数
            on_retry: 重试时的回调函数 (attempt, error, delay)
            *args: 传递给 fn 的位置参数
            **kwargs: 传递给 fn 的关键字参数

        Returns:
            fn 的返回值

        Raises:
            Exception: 重试耗尽或不可重试的异常
        """
        last_error: Exception | None = None

        for attempt in range(self.config.max_retries + 1):
            try:
                return await fn(*args, **kwargs)
            except Exception as e:
                last_error = e

                # 不可重试的错误立即抛出
                if not self._is_retryable(e):
                    raise

                # 已耗尽重试次数
                if attempt >= self.config.max_retries:
                    raise

                # 计算延迟并等待
                delay = self._calculate_delay(attempt)
                if on_retry:
                    on_retry(attempt + 1, e, delay)
                await asyncio.sleep(delay)

        raise last_error  # type: ignore[misc]
