"""重试机制测试"""

import pytest
from pure_agent_loop.retry import RetryConfig, RetryHandler


class TestRetryConfig:
    """RetryConfig 测试"""

    def test_default_values(self):
        """应该有合理的默认值"""
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay == 1.0
        assert config.max_delay == 30.0

    def test_custom_values(self):
        """应该支持自定义值"""
        config = RetryConfig(max_retries=5, base_delay=0.5, max_delay=10.0)
        assert config.max_retries == 5


class TestRetryHandler:
    """RetryHandler 测试"""

    @pytest.mark.asyncio
    async def test_success_no_retry(self):
        """成功调用不应重试"""
        call_count = 0

        async def success_fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        handler = RetryHandler(RetryConfig())
        result = await handler.execute(success_fn)
        assert result == "ok"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_retry_on_failure(self):
        """失败后应该重试"""
        call_count = 0

        async def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ConnectionError("连接超时")
            return "ok"

        handler = RetryHandler(RetryConfig(base_delay=0.01))
        result = await handler.execute(flaky_fn)
        assert result == "ok"
        assert call_count == 3

    @pytest.mark.asyncio
    async def test_max_retries_exceeded(self):
        """超过最大重试次数应该抛出异常"""

        async def always_fail():
            raise ConnectionError("连接超时")

        handler = RetryHandler(RetryConfig(max_retries=2, base_delay=0.01))
        with pytest.raises(ConnectionError):
            await handler.execute(always_fail)

    @pytest.mark.asyncio
    async def test_non_retryable_error(self):
        """不可重试的错误应该立即抛出"""
        call_count = 0

        async def bad_fn():
            nonlocal call_count
            call_count += 1
            raise ValueError("参数错误")

        handler = RetryHandler(RetryConfig(base_delay=0.01))
        with pytest.raises(ValueError):
            await handler.execute(bad_fn)
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_on_retry_callback(self):
        """应该在每次重试时调用回调"""
        retry_events = []
        call_count = 0

        async def flaky_fn():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ConnectionError("连接超时")
            return "ok"

        def on_retry(attempt, error, delay):
            retry_events.append((attempt, str(error), delay))

        handler = RetryHandler(RetryConfig(base_delay=0.01))
        result = await handler.execute(flaky_fn, on_retry=on_retry)
        assert result == "ok"
        assert len(retry_events) == 1
        assert retry_events[0][0] == 1

    def test_calculate_delay(self):
        """延迟应该使用指数退避"""
        handler = RetryHandler(RetryConfig(base_delay=1.0, max_delay=30.0))
        assert handler._calculate_delay(0) == 1.0
        assert handler._calculate_delay(1) == 2.0
        assert handler._calculate_delay(2) == 4.0
        assert handler._calculate_delay(10) == 30.0  # 不超过 max_delay
