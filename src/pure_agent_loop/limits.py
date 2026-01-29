"""终止控制

管理 Agentic Loop 的终止条件，区分软限制（通知 AI 调整）和硬限制（强制终止）。
"""

import time
from dataclasses import dataclass, field


@dataclass
class LoopLimits:
    """循环限制配置

    Attributes:
        max_steps: 软限制 - 每 N 步触发一次检查点（默认 10）
        timeout: 软限制 - 每 N 秒触发一次检查点（默认 300 秒）
        max_tokens: 硬限制 - 累计 token 上限（默认 100,000）
        step_limit_prompt: 步数软限制的提示模板
        timeout_prompt: 超时软限制的提示模板
    """

    max_steps: int = 10
    timeout: float = 300.0
    max_tokens: int = 100_000

    step_limit_prompt: str = (
        "⚠️ 你已完成第 {checkpoint_steps} 步检查点（总步数: {current_steps}）。\n"
        "请评估当前进度:\n"
        "- 如果接近完成，请尽快总结答案\n"
        "- 如果陷入循环，请调整策略或尝试其他方法\n"
        "- 如果需要更多步骤，请继续但保持高效"
    )
    timeout_prompt: str = (
        "⚠️ 执行已超时（已用 {elapsed:.1f}s / 每轮上限 {timeout}s），"
        "请评估当前进度，尽快总结已获得的信息或调整策略。"
    )


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
    软限制使用周期性检查点机制，每 N 步/秒触发一次。

    Args:
        limits: 限制配置
    """

    def __init__(self, limits: LoopLimits):
        self.limits = limits
        self.start_time = time.time()
        self.current_step: int = 0
        self.total_tokens: int = 0
        self._step_checkpoint: int = 0
        self._timeout_checkpoint: float = 0.0

    def add_tokens(self, tokens: int) -> None:
        """累加 token 用量"""
        self.total_tokens += tokens

    @property
    def elapsed(self) -> float:
        """已用时间（秒）"""
        return time.time() - self.start_time

    def check(self) -> LimitResult:
        """检查终止条件

        Returns:
            LimitResult: 检查结果，包含行动指令和提示
        """
        # 硬限制优先 - token 上限
        if self.total_tokens >= self.limits.max_tokens:
            return LimitResult.stop("token_limit")

        # 软限制：步数周期检查
        steps_since_checkpoint = self.current_step - self._step_checkpoint
        if steps_since_checkpoint >= self.limits.max_steps:
            self._step_checkpoint = self.current_step
            prompt = self.limits.step_limit_prompt.format(
                checkpoint_steps=self.limits.max_steps,
                current_steps=self.current_step,
            )
            return LimitResult.warn("step_limit", prompt)

        # 软限制：超时周期检查
        elapsed = self.elapsed
        if elapsed - self._timeout_checkpoint >= self.limits.timeout:
            self._timeout_checkpoint = elapsed
            prompt = self.limits.timeout_prompt.format(
                elapsed=elapsed,
                timeout=self.limits.timeout,
            )
            return LimitResult.warn("timeout", prompt)

        return LimitResult.continue_running()
