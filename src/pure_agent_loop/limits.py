"""终止控制

管理 Agentic Loop 的终止条件。
支持软限制（注入提示继续执行）和硬限制（禁用工具要求总结）双模式。
"""

import json
from collections import deque
from dataclasses import dataclass, field


# 软限制默认提示词
DEFAULT_SOFT_LIMIT_PROMPT = (
    "你已执行了 {current_steps} 步。请评估当前进度:\n"
    "- 如果接近完成，请尽快总结答案\n"
    "- 如果陷入循环，请调整策略或尝试其他方法\n"
    "- 如果需要更多步骤，请继续但保持高效"
)

# 硬限制默认提示词（参考 opencode max-steps.txt）
DEFAULT_HARD_LIMIT_PROMPT = (
    "已达到最大步数限制。工具已禁用，请仅使用文本回复。\n\n"
    "请在回复中包含:\n"
    "- 已完成的工作总结\n"
    "- 未完成的任务列表\n"
    "- 后续建议"
)


@dataclass
class LoopLimits:
    """循环限制配置

    Attributes:
        max_steps: 步数上限，None 表示无限制
        step_limit_mode: 步数限制模式 - "soft"（注入提示继续）或 "hard"（禁用工具终止）
        max_tokens: 累计生成 token 上限（仅计 completion_tokens，始终为硬限制）
        doom_loop_threshold: 连续相同工具调用的检测阈值
        soft_limit_prompt: 软限制提示词模板（支持 {current_steps} 占位符）
        soft_limit_interval: 软模式下首次触发后的重复提醒间隔（默认等于 max_steps）
        hard_limit_prompt: 硬限制提示词（达到上限时注入）
    """

    max_steps: int | None = None
    step_limit_mode: str = "soft"
    max_tokens: int = 200_000
    doom_loop_threshold: int = 3

    soft_limit_prompt: str = DEFAULT_SOFT_LIMIT_PROMPT
    soft_limit_interval: int | None = None
    hard_limit_prompt: str = DEFAULT_HARD_LIMIT_PROMPT

    @property
    def effective_soft_interval(self) -> int | None:
        """实际的软限制重复间隔"""
        if self.soft_limit_interval is not None:
            return self.soft_limit_interval
        return self.max_steps


@dataclass
class LimitResult:
    """限制检查结果

    Attributes:
        action: 行动指令 - "continue" | "warn" | "stop"
        reason: 触发原因
        prompt: 注入给 AI 的提示（仅 warn 时有值）
    """

    action: str
    reason: str = ""
    prompt: str = ""

    @classmethod
    def continue_running(cls) -> "LimitResult":
        """创建继续执行的结果"""
        return cls(action="continue")

    @classmethod
    def warn(cls, reason: str, prompt: str) -> "LimitResult":
        """创建软限制警告结果"""
        return cls(action="warn", reason=reason, prompt=prompt)

    @classmethod
    def stop(cls, reason: str) -> "LimitResult":
        """创建硬限制停止结果"""
        return cls(action="stop", reason=reason)


class LimitChecker:
    """限制检查器

    每步结束后调用 check() 判断是否触发限制。

    Args:
        limits: 限制配置
    """

    def __init__(self, limits: LoopLimits):
        self.limits = limits
        self.current_step: int = 0
        self.total_tokens: int = 0
        self._last_warn_step: int = 0
        # doom loop 检测：记录最近 N 次工具调用签名
        self._recent_calls: deque[str] = deque(
            maxlen=limits.doom_loop_threshold
        )

    def add_tokens(self, tokens: int) -> None:
        """累加生成 token 用量（仅 completion_tokens）"""
        self.total_tokens += tokens

    def record_tool_call(self, tool_name: str, arguments_json: str) -> None:
        """记录一次工具调用（用于 doom loop 检测）"""
        signature = f"{tool_name}:{arguments_json}"
        self._recent_calls.append(signature)

    @property
    def is_doom_loop(self) -> bool:
        """是否检测到 doom loop"""
        threshold = self.limits.doom_loop_threshold
        if len(self._recent_calls) < threshold:
            return False
        # 检查最近 N 次调用是否完全相同
        calls = list(self._recent_calls)
        return len(set(calls[-threshold:])) == 1

    @property
    def is_last_step(self) -> bool:
        """是否为最后一步（仅硬模式有效）

        用于循环体判断是否在下次 LLM 调用时禁用工具。
        """
        if self.limits.max_steps is None:
            return False
        if self.limits.step_limit_mode != "hard":
            return False
        return self.current_step >= self.limits.max_steps

    def check(self) -> LimitResult:
        """检查终止条件

        优先级：token 硬限制 > 步数限制（软/硬）
        """
        # 硬限制优先：token 上限
        if self.total_tokens >= self.limits.max_tokens:
            return LimitResult.stop("token_limit")

        # 步数限制
        if self.limits.max_steps is not None and self.current_step >= self.limits.max_steps:
            if self.limits.step_limit_mode == "hard":
                return LimitResult.stop("max_steps")
            else:
                # 软模式：检查是否到达提醒时机
                interval = self.limits.effective_soft_interval or self.limits.max_steps
                if self._last_warn_step == 0:
                    # 首次触发
                    self._last_warn_step = self.current_step
                    prompt = self.limits.soft_limit_prompt.format(
                        current_steps=self.current_step,
                    )
                    return LimitResult.warn("step_limit", prompt)
                elif self.current_step - self._last_warn_step >= interval:
                    # 周期性重复提醒
                    self._last_warn_step = self.current_step
                    prompt = self.limits.soft_limit_prompt.format(
                        current_steps=self.current_step,
                    )
                    return LimitResult.warn("step_limit", prompt)

        return LimitResult.continue_running()
