"""终止控制测试"""

import json
import pytest
from pure_agent_loop.limits import LoopLimits, LimitChecker, LimitResult


class TestLoopLimits:
    """LoopLimits 配置测试"""

    def test_default_values(self):
        """默认值：无步数限制、软模式、200k token"""
        limits = LoopLimits()
        assert limits.max_steps is None
        assert limits.step_limit_mode == "soft"
        assert limits.max_tokens == 200_000
        assert limits.doom_loop_threshold == 3

    def test_custom_values(self):
        """支持自定义所有字段"""
        limits = LoopLimits(
            max_steps=20,
            step_limit_mode="hard",
            max_tokens=50_000,
            doom_loop_threshold=5,
        )
        assert limits.max_steps == 20
        assert limits.step_limit_mode == "hard"
        assert limits.max_tokens == 50_000
        assert limits.doom_loop_threshold == 5

    def test_soft_limit_interval_defaults_to_max_steps(self):
        """soft_limit_interval 未设置时应等于 max_steps"""
        limits = LoopLimits(max_steps=10)
        assert limits.effective_soft_interval == 10

    def test_soft_limit_interval_custom(self):
        """自定义 soft_limit_interval"""
        limits = LoopLimits(max_steps=10, soft_limit_interval=5)
        assert limits.effective_soft_interval == 5

    def test_has_default_prompts(self):
        """应包含默认的软/硬限制提示词"""
        limits = LoopLimits()
        assert len(limits.soft_limit_prompt) > 0
        assert len(limits.hard_limit_prompt) > 0


class TestLimitResult:
    """LimitResult 测试"""

    def test_continue_result(self):
        result = LimitResult.continue_running()
        assert result.action == "continue"

    def test_warn_result(self):
        result = LimitResult.warn("step_limit", "请调整策略")
        assert result.action == "warn"
        assert result.reason == "step_limit"
        assert result.prompt == "请调整策略"

    def test_stop_result(self):
        result = LimitResult.stop("token_limit")
        assert result.action == "stop"
        assert result.reason == "token_limit"


class TestLimitCheckerTokenLimit:
    """Token 硬限制测试"""

    def test_token_hard_limit_stops(self):
        """token 超限应触发硬停止"""
        checker = LimitChecker(LoopLimits(max_tokens=1000))
        checker.add_tokens(1500)
        checker.current_step = 1
        result = checker.check()
        assert result.action == "stop"
        assert result.reason == "token_limit"

    def test_add_tokens_accumulates(self):
        """token 应累加"""
        checker = LimitChecker(LoopLimits())
        checker.add_tokens(100)
        checker.add_tokens(200)
        assert checker.total_tokens == 300

    def test_token_limit_takes_priority(self):
        """token 硬限制优先于步数限制"""
        checker = LimitChecker(LoopLimits(max_steps=3, max_tokens=100))
        checker.current_step = 3
        checker.add_tokens(200)
        result = checker.check()
        assert result.action == "stop"
        assert result.reason == "token_limit"


class TestLimitCheckerSoftMode:
    """软限制模式测试"""

    def test_no_limit_when_max_steps_none(self):
        """max_steps=None 时不触发步数限制"""
        checker = LimitChecker(LoopLimits(max_steps=None))
        checker.current_step = 100
        result = checker.check()
        assert result.action == "continue"

    def test_soft_limit_warns_at_threshold(self):
        """软模式到达 max_steps 应产出 warn"""
        checker = LimitChecker(LoopLimits(max_steps=3, step_limit_mode="soft"))
        checker.current_step = 3
        result = checker.check()
        assert result.action == "warn"
        assert result.reason == "step_limit"

    def test_soft_limit_repeats_at_interval(self):
        """软模式应按 interval 周期重复提醒"""
        checker = LimitChecker(LoopLimits(max_steps=3, step_limit_mode="soft"))
        # 第 3 步触发
        checker.current_step = 3
        assert checker.check().action == "warn"
        # 第 4、5 步不触发
        checker.current_step = 4
        assert checker.check().action == "continue"
        checker.current_step = 5
        assert checker.check().action == "continue"
        # 第 6 步再次触发（interval=max_steps=3）
        checker.current_step = 6
        assert checker.check().action == "warn"

    def test_soft_limit_custom_interval(self):
        """自定义 interval"""
        limits = LoopLimits(max_steps=5, step_limit_mode="soft", soft_limit_interval=2)
        checker = LimitChecker(limits)
        # 第 5 步首次触发
        checker.current_step = 5
        assert checker.check().action == "warn"
        # 第 6 步不触发
        checker.current_step = 6
        assert checker.check().action == "continue"
        # 第 7 步触发（5+2=7）
        checker.current_step = 7
        assert checker.check().action == "warn"

    def test_soft_limit_prompt_formatted(self):
        """软限制提示词应包含步数"""
        checker = LimitChecker(LoopLimits(max_steps=5, step_limit_mode="soft"))
        checker.current_step = 5
        result = checker.check()
        assert "5" in result.prompt


class TestLimitCheckerHardMode:
    """硬限制模式测试"""

    def test_hard_limit_stops_at_threshold(self):
        """硬模式到达 max_steps 应产出 stop"""
        checker = LimitChecker(LoopLimits(max_steps=3, step_limit_mode="hard"))
        checker.current_step = 3
        result = checker.check()
        assert result.action == "stop"
        assert result.reason == "max_steps"

    def test_hard_limit_continues_before_threshold(self):
        """硬模式未达阈值应继续"""
        checker = LimitChecker(LoopLimits(max_steps=3, step_limit_mode="hard"))
        checker.current_step = 2
        result = checker.check()
        assert result.action == "continue"

    def test_is_last_step_property(self):
        """is_last_step 属性：硬模式下到达阈值前一步"""
        checker = LimitChecker(LoopLimits(max_steps=5, step_limit_mode="hard"))
        checker.current_step = 4
        assert not checker.is_last_step
        checker.current_step = 5
        assert checker.is_last_step

    def test_is_last_step_soft_mode_always_false(self):
        """软模式下 is_last_step 始终为 False"""
        checker = LimitChecker(LoopLimits(max_steps=5, step_limit_mode="soft"))
        checker.current_step = 5
        assert not checker.is_last_step

    def test_is_last_step_no_limit(self):
        """无步数限制时 is_last_step 始终为 False"""
        checker = LimitChecker(LoopLimits(max_steps=None))
        checker.current_step = 999
        assert not checker.is_last_step


class TestLimitCheckerDoomLoop:
    """Doom loop 检测测试"""

    def test_no_doom_loop_initially(self):
        """初始状态不应检测到 doom loop"""
        checker = LimitChecker(LoopLimits(doom_loop_threshold=3))
        assert not checker.is_doom_loop

    def test_doom_loop_detected(self):
        """连续相同工具调用应触发 doom loop"""
        checker = LimitChecker(LoopLimits(doom_loop_threshold=3))
        call = ("search", json.dumps({"query": "test"}))
        checker.record_tool_call(call[0], call[1])
        checker.record_tool_call(call[0], call[1])
        checker.record_tool_call(call[0], call[1])
        assert checker.is_doom_loop

    def test_no_doom_loop_different_calls(self):
        """不同的工具调用不触发 doom loop"""
        checker = LimitChecker(LoopLimits(doom_loop_threshold=3))
        checker.record_tool_call("search", json.dumps({"query": "a"}))
        checker.record_tool_call("search", json.dumps({"query": "b"}))
        checker.record_tool_call("search", json.dumps({"query": "c"}))
        assert not checker.is_doom_loop

    def test_doom_loop_broken_by_different_call(self):
        """中间插入不同调用应重置 doom loop 计数"""
        checker = LimitChecker(LoopLimits(doom_loop_threshold=3))
        call = ("search", json.dumps({"query": "test"}))
        checker.record_tool_call(call[0], call[1])
        checker.record_tool_call(call[0], call[1])
        checker.record_tool_call("read", json.dumps({"file": "a.txt"}))  # 打断
        checker.record_tool_call(call[0], call[1])
        assert not checker.is_doom_loop

    def test_doom_loop_custom_threshold(self):
        """自定义阈值"""
        checker = LimitChecker(LoopLimits(doom_loop_threshold=2))
        call = ("search", json.dumps({"query": "test"}))
        checker.record_tool_call(call[0], call[1])
        assert not checker.is_doom_loop
        checker.record_tool_call(call[0], call[1])
        assert checker.is_doom_loop
