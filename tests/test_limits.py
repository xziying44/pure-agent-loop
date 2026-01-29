"""终止控制测试"""

import time
import pytest
from pure_agent_loop.limits import LoopLimits, LimitChecker, LimitResult


class TestLoopLimits:
    """LoopLimits 配置测试"""

    def test_default_values(self):
        """应该有合理的默认值"""
        limits = LoopLimits()
        assert limits.max_steps == 10
        assert limits.timeout == 300.0
        assert limits.max_tokens == 100_000

    def test_custom_values(self):
        """应该支持自定义值"""
        limits = LoopLimits(max_steps=5, timeout=60.0, max_tokens=50_000)
        assert limits.max_steps == 5
        assert limits.timeout == 60.0
        assert limits.max_tokens == 50_000


class TestLimitResult:
    """LimitResult 测试"""

    def test_continue_result(self):
        """继续执行的结果"""
        result = LimitResult.continue_running()
        assert result.action == "continue"
        assert result.reason == ""
        assert result.prompt == ""

    def test_warn_result(self):
        """软限制警告结果"""
        result = LimitResult.warn("step_limit", "请调整策略")
        assert result.action == "warn"
        assert result.reason == "step_limit"
        assert result.prompt == "请调整策略"

    def test_stop_result(self):
        """硬限制停止结果"""
        result = LimitResult.stop("token_limit")
        assert result.action == "stop"
        assert result.reason == "token_limit"


class TestLimitChecker:
    """LimitChecker 测试"""

    def test_continue_within_limits(self):
        """在限制范围内应该返回 continue"""
        checker = LimitChecker(LoopLimits(max_steps=10))
        checker.current_step = 5
        result = checker.check()
        assert result.action == "continue"

    def test_step_soft_limit_triggers(self):
        """达到步数限制应触发软限制"""
        checker = LimitChecker(LoopLimits(max_steps=3))
        checker.current_step = 3
        result = checker.check()
        assert result.action == "warn"
        assert result.reason == "step_limit"

    def test_step_soft_limit_resets(self):
        """软限制触发后应重置检查点，下一个周期再触发"""
        checker = LimitChecker(LoopLimits(max_steps=3))

        # 第 3 步触发
        checker.current_step = 3
        result = checker.check()
        assert result.action == "warn"

        # 第 4、5 步不触发
        checker.current_step = 4
        assert checker.check().action == "continue"
        checker.current_step = 5
        assert checker.check().action == "continue"

        # 第 6 步再次触发
        checker.current_step = 6
        result = checker.check()
        assert result.action == "warn"
        assert result.reason == "step_limit"

    def test_token_hard_limit(self):
        """token 超限应触发硬限制"""
        checker = LimitChecker(LoopLimits(max_tokens=1000))
        checker.total_tokens = 1500
        result = checker.check()
        assert result.action == "stop"
        assert result.reason == "token_limit"

    def test_hard_limit_takes_priority(self):
        """硬限制应该优先于软限制"""
        checker = LimitChecker(LoopLimits(max_steps=3, max_tokens=1000))
        checker.current_step = 3
        checker.total_tokens = 1500
        result = checker.check()
        assert result.action == "stop"
        assert result.reason == "token_limit"

    def test_timeout_soft_limit(self):
        """超时应触发软限制"""
        checker = LimitChecker(LoopLimits(timeout=0.01))
        time.sleep(0.02)
        checker.current_step = 1
        result = checker.check()
        assert result.action == "warn"
        assert result.reason == "timeout"

    def test_add_tokens(self):
        """应该能累加 token"""
        checker = LimitChecker(LoopLimits())
        checker.add_tokens(100)
        assert checker.total_tokens == 100
        checker.add_tokens(200)
        assert checker.total_tokens == 300

    def test_step_limit_prompt_formatted(self):
        """软限制提示应包含格式化的步数"""
        checker = LimitChecker(LoopLimits(max_steps=5))
        checker.current_step = 5
        result = checker.check()
        assert "5" in result.prompt
