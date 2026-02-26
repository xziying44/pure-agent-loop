# 系统提示词 + Todo + 循环体 + Skill 引导 全面重构实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 参考 opencode 的成熟设计，重构 pure-agent-loop 的系统提示词、Todo 工具、循环体限制机制和 Skill 引导，使其更专业、更灵活。

**Architecture:** 保持现有模块分层不变（agent → loop → limits/tools）。提示词从"SOP 强制协议"改为 opencode 风格的温和引导。循环体引入软/硬双模式步数限制和 doom loop 检测。Todo 工具扩展为四状态。

**Tech Stack:** Python 3.10+, pytest, pytest-asyncio

**设计文档:** `docs/plans/2026-02-26-system-refactor-design.md`

---

### Task 1: 重构 limits.py — LoopLimits 和 LimitChecker

**Files:**
- Modify: `src/pure_agent_loop/limits.py`
- Test: `tests/test_limits.py`

**Step 1: 重写 test_limits.py 测试**

完全重写测试文件，覆盖新的 LoopLimits 接口和双模式限制逻辑：

```python
"""终止控制测试"""

import json
import pytest
from pure_agent_loop.limits import LoopLimits, LimitChecker, LimitResult


class TestLoopLimits:
    """LoopLimits 配置测试"""

    def test_default_values(self):
        """默认值：无步数限制、软模式、100k token"""
        limits = LoopLimits()
        assert limits.max_steps is None
        assert limits.step_limit_mode == "soft"
        assert limits.max_tokens == 100_000
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
```

**Step 2: 运行测试确认全部失败**

Run: `source venv/bin/activate && pytest tests/test_limits.py -v`
Expected: 全部 FAIL（LoopLimits 接口变了）

**Step 3: 重写 limits.py 实现**

```python
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
        max_tokens: 累计 token 上限（始终为硬限制）
        doom_loop_threshold: 连续相同工具调用的检测阈值
        soft_limit_prompt: 软限制提示词模板（支持 {current_steps} 占位符）
        soft_limit_interval: 软模式下首次触发后的重复提醒间隔（默认等于 max_steps）
        hard_limit_prompt: 硬限制提示词（达到上限时注入）
    """

    max_steps: int | None = None
    step_limit_mode: str = "soft"
    max_tokens: int = 100_000
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
        """累加 token 用量"""
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
```

**Step 4: 运行测试确认全部通过**

Run: `source venv/bin/activate && pytest tests/test_limits.py -v`
Expected: 全部 PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/limits.py tests/test_limits.py
git commit -m "refactor(limits): 双模式步数限制 + doom loop 检测，去掉超时机制"
```

---

### Task 2: 重构 builtin_tools.py — Todo 四状态 + priority

**Files:**
- Modify: `src/pure_agent_loop/builtin_tools.py`
- Test: `tests/test_builtin_tools.py`

**Step 1: 重写 test_builtin_tools.py 测试**

```python
"""内置工具测试"""

import pytest
from pure_agent_loop.builtin_tools import (
    TodoItem, TodoStore, create_todo_tool, VALID_STATUSES, VALID_PRIORITIES,
)
from pure_agent_loop.tool import Tool


class TestTodoItem:
    """TodoItem 数据类测试"""

    def test_create_default(self):
        """默认状态为 pending、优先级为 medium"""
        item = TodoItem(content="测试任务")
        assert item.status == "pending"
        assert item.priority == "medium"

    def test_all_valid_statuses(self):
        """四种状态都应合法"""
        for status in ("pending", "in_progress", "completed", "cancelled"):
            item = TodoItem(content="任务", status=status)
            assert item.status == status

    def test_all_valid_priorities(self):
        """三种优先级都应合法"""
        for priority in ("high", "medium", "low"):
            item = TodoItem(content="任务", priority=priority)
            assert item.priority == priority

    def test_invalid_status_raises_error(self):
        """无效状态应抛出 ValueError"""
        with pytest.raises(ValueError, match="无效的任务状态"):
            TodoItem(content="测试", status="unknown")

    def test_invalid_priority_raises_error(self):
        """无效优先级应抛出 ValueError"""
        with pytest.raises(ValueError, match="无效的优先级"):
            TodoItem(content="测试", priority="urgent")

    def test_to_dict_includes_priority(self):
        """to_dict 应包含 priority 字段"""
        item = TodoItem(content="A", status="in_progress", priority="high")
        d = item.to_dict()
        assert d == {"content": "A", "status": "in_progress", "priority": "high"}

    def test_valid_statuses_constant(self):
        """VALID_STATUSES 应包含四种状态"""
        assert VALID_STATUSES == ("pending", "in_progress", "completed", "cancelled")

    def test_valid_priorities_constant(self):
        """VALID_PRIORITIES 应包含三种优先级"""
        assert VALID_PRIORITIES == ("high", "medium", "low")


class TestTodoStore:
    """TodoStore 测试"""

    def test_initial_empty(self):
        store = TodoStore()
        assert store.todos == []

    def test_write_with_in_progress(self):
        """支持 in_progress 状态"""
        store = TodoStore()
        store.write([
            {"content": "正在做", "status": "in_progress"},
        ])
        assert store.todos[0].status == "in_progress"

    def test_write_with_cancelled(self):
        """支持 cancelled 状态"""
        store = TodoStore()
        store.write([{"content": "已取消", "status": "cancelled"}])
        assert store.todos[0].status == "cancelled"

    def test_write_with_priority(self):
        """支持 priority 字段"""
        store = TodoStore()
        store.write([{"content": "紧急任务", "status": "pending", "priority": "high"}])
        assert store.todos[0].priority == "high"

    def test_write_default_priority(self):
        """未指定 priority 时默认 medium"""
        store = TodoStore()
        store.write([{"content": "任务", "status": "pending"}])
        assert store.todos[0].priority == "medium"

    def test_write_replaces_list(self):
        store = TodoStore()
        store.write([
            {"content": "A", "status": "pending"},
            {"content": "B", "status": "completed"},
        ])
        assert len(store.todos) == 2

    def test_write_returns_formatted_string(self):
        store = TodoStore()
        result = store.write([
            {"content": "搜索", "status": "completed"},
            {"content": "分析", "status": "in_progress"},
            {"content": "报告", "status": "pending"},
        ])
        assert "✅" in result
        assert "🔵" in result
        assert "⬜" in result

    def test_write_empty_list(self):
        store = TodoStore()
        store.write([{"content": "A", "status": "pending"}])
        result = store.write([])
        assert store.todos == []
        assert "空" in result

    def test_write_invalid_status_returns_error(self):
        """写入无效状态应返回错误，保持旧列表"""
        store = TodoStore()
        store.write([{"content": "旧任务", "status": "pending"}])
        result = store.write([{"content": "新任务", "status": "bad"}])
        assert "失败" in result
        assert len(store.todos) == 1
        assert store.todos[0].content == "旧任务"

    def test_todos_property_returns_copy(self):
        store = TodoStore()
        store.write([{"content": "A", "status": "pending"}])
        store.todos.clear()
        assert len(store.todos) == 1

    def test_format_output_summary_four_statuses(self):
        """格式化输出应显示四种状态的统计"""
        store = TodoStore()
        result = store.write([
            {"content": "A", "status": "pending"},
            {"content": "B", "status": "in_progress"},
            {"content": "C", "status": "completed"},
            {"content": "D", "status": "cancelled"},
        ])
        assert "待处理: 1" in result
        assert "进行中: 1" in result
        assert "已完成: 1" in result
        assert "已取消: 1" in result


class TestCreateTodoTool:
    """create_todo_tool 工厂函数测试"""

    def test_creates_tool_instance(self):
        store = TodoStore()
        t = create_todo_tool(store)
        assert isinstance(t, Tool)
        assert t.name == "todo_write"

    def test_tool_schema_has_four_statuses(self):
        """schema 中 status 应有四种枚举值"""
        store = TodoStore()
        t = create_todo_tool(store)
        schema = t.to_openai_schema()
        status_prop = schema["function"]["parameters"]["properties"]["todos"]["items"]["properties"]["status"]
        assert set(status_prop["enum"]) == {"pending", "in_progress", "completed", "cancelled"}

    def test_tool_schema_has_priority(self):
        """schema 中应包含 priority 字段"""
        store = TodoStore()
        t = create_todo_tool(store)
        schema = t.to_openai_schema()
        item_props = schema["function"]["parameters"]["properties"]["todos"]["items"]["properties"]
        assert "priority" in item_props

    async def test_tool_execute_updates_store(self):
        store = TodoStore()
        t = create_todo_tool(store)
        await t.execute({
            "todos": [
                {"content": "A", "status": "in_progress", "priority": "high"},
                {"content": "B", "status": "pending"},
            ]
        })
        assert len(store.todos) == 2
        assert store.todos[0].priority == "high"
        assert store.todos[1].priority == "medium"  # 默认值

    def test_tool_description_length(self):
        """工具描述应足够详细（参考 opencode 的 167 行描述）"""
        store = TodoStore()
        t = create_todo_tool(store)
        assert len(t.description) > 200
```

**Step 2: 运行测试确认失败**

Run: `source venv/bin/activate && pytest tests/test_builtin_tools.py -v`
Expected: 大部分 FAIL

**Step 3: 重写 builtin_tools.py 实现**

完整重写 `builtin_tools.py`，扩展四状态 + priority + 重写工具描述。

工具描述参照 opencode `todowrite.txt` 翻译并适配通用场景。关键内容包括：
- 何时使用（3+ 步骤的复杂任务、用户提供多个任务、需要规划的任务）
- 何时不使用（单步简单任务、纯对话、不需要追踪的任务）
- 任务状态说明（pending/in_progress/completed/cancelled）
- 管理规则（一次一个 in_progress、立即标记完成）

`TodoItem` 新增 `priority` 字段和校验。`TodoStore._format_output` 更新四状态的图标和统计。`create_todo_tool` 更新 schema 和描述。

**Step 4: 运行测试确认通过**

Run: `source venv/bin/activate && pytest tests/test_builtin_tools.py -v`
Expected: 全部 PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/builtin_tools.py tests/test_builtin_tools.py
git commit -m "refactor(todo): 四状态(pending/in_progress/completed/cancelled) + priority + 重写工具描述"
```

---

### Task 3: 重写 prompts.py — 系统提示词

**Files:**
- Modify: `src/pure_agent_loop/prompts.py`
- Test: `tests/test_prompts.py`

**Step 1: 重写 test_prompts.py 测试**

```python
"""系统提示词测试"""

from pure_agent_loop.prompts import build_system_prompt


class TestBuildSystemPrompt:
    """build_system_prompt 测试"""

    def test_default_name(self):
        """默认名称应为 '智能助理'"""
        prompt = build_system_prompt()
        assert "智能助理" in prompt

    def test_custom_name(self):
        """自定义名称应注入"""
        prompt = build_system_prompt(name="研究助手")
        assert "研究助手" in prompt

    def test_user_prompt_injected(self):
        """用户提示词应注入"""
        prompt = build_system_prompt(user_prompt="你擅长数学。")
        assert "你擅长数学。" in prompt

    def test_empty_user_prompt(self):
        """空用户提示词不应异常"""
        prompt = build_system_prompt(user_prompt="")
        assert "智能助理" in prompt

    def test_contains_objectivity_section(self):
        """应包含专业客观性段落"""
        prompt = build_system_prompt()
        assert "客观" in prompt

    def test_contains_todo_guidance(self):
        """应包含 todo_write 使用引导"""
        prompt = build_system_prompt()
        assert "todo_write" in prompt

    def test_contains_skill_guidance(self):
        """应包含技能系统引导"""
        prompt = build_system_prompt()
        assert "技能" in prompt or "skill" in prompt.lower()

    def test_contains_tool_usage_guidance(self):
        """应包含工具使用策略"""
        prompt = build_system_prompt()
        assert "并行" in prompt

    def test_contains_output_style_guidance(self):
        """应包含输出风格指南"""
        prompt = build_system_prompt()
        assert "简洁" in prompt

    def test_no_mandatory_protocol_language(self):
        """不应包含旧的强制协议用语"""
        prompt = build_system_prompt()
        assert "唯一起手式" not in prompt
        assert "SOP" not in prompt
        assert "严禁" not in prompt
        assert "综合研判" not in prompt

    def test_user_section_at_end(self):
        """用户自定义指令应在提示词末尾"""
        prompt = build_system_prompt(user_prompt="自定义内容")
        # 用户内容应在提示词的后半部分
        idx = prompt.index("自定义内容")
        assert idx > len(prompt) // 2
```

**Step 2: 运行测试确认失败**

Run: `source venv/bin/activate && pytest tests/test_prompts.py -v`
Expected: 多数 FAIL

**Step 3: 重写 prompts.py 实现**

完全重写提示词内容，保持 `build_system_prompt(name, user_prompt)` 接口不变。新的 6 段结构：

1. **角色定义** — 通用智能助理定位
2. **专业客观性** — 参考 opencode `anthropic.txt` 中的 Professional objectivity 段
3. **任务管理引导** — opencode 风格温和引导，引用 `todo_write` 工具
4. **技能系统引导** — 温和版，简洁的何时用/何时不用
5. **工具使用策略** — 并行独立工具，优先专用工具
6. **输出风格指南** — 简洁协作，默认做工作不问"要不要继续"，用户自定义注入区

**Step 4: 运行测试确认通过**

Run: `source venv/bin/activate && pytest tests/test_prompts.py -v`
Expected: 全部 PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/prompts.py tests/test_prompts.py
git commit -m "refactor(prompts): 系统提示词重写为 opencode 风格温和引导，6 段结构"
```

---

### Task 4: 重构 loop.py — 双模式限制 + doom loop

**Files:**
- Modify: `src/pure_agent_loop/loop.py`
- Test: `tests/test_loop.py`

**Step 1: 在 test_loop.py 中新增/修改测试**

保留现有测试中仍然有效的部分（简单文本响应、工具调用、并行执行等），修改步数限制相关测试，新增 doom loop 和硬限制测试。

需要修改的测试：
- `test_step_soft_limit_triggers` → 适配新的 LoopLimits 接口（`max_steps=3, step_limit_mode="soft"`）

需要新增的测试：
```python
class TestReactLoopHardLimit:
    """硬限制模式测试"""

    async def test_hard_limit_disables_tools_and_stops(self):
        """硬模式达到 max_steps 后，LLM 应无工具调用，循环终止"""
        # 构造 MockLLM：
        # step 1-2: 正常工具调用
        # step 3（最后一步）: 因为 tools=None，LLM 返回纯文本总结
        # 验证 stop_reason == "max_steps"

    async def test_hard_limit_injects_prompt(self):
        """硬模式最后一步应注入 hard_limit_prompt"""
        # 通过检查 MockLLM 收到的 messages 中是否包含 hard_limit_prompt


class TestReactLoopDoomLoop:
    """Doom loop 检测测试"""

    async def test_doom_loop_terminates(self):
        """连续相同工具调用应触发 doom loop 终止"""
        # 构造 MockLLM：连续 3 次返回完全相同的 tool_call
        # 验证：产出 ERROR 事件，stop_reason == "doom_loop"

    async def test_no_doom_loop_with_different_args(self):
        """不同参数的工具调用不触发 doom loop"""
        # 构造 MockLLM：3 次相同工具名但不同参数
        # 验证：正常完成
```

**Step 2: 运行测试确认新增测试失败**

Run: `source venv/bin/activate && pytest tests/test_loop.py -v`
Expected: 新增测试 FAIL，原有有效测试可能 PASS 或 FAIL（取决于 LoopLimits 接口变化）

**Step 3: 修改 loop.py 实现**

关键修改点：

1. **在循环开始前**：判断 `checker.is_last_step`，如果为 True 则在下次 LLM 调用时传 `tools=None` 并注入 `hard_limit_prompt`

2. **工具执行后**：调用 `checker.record_tool_call(name, json.dumps(args))` 记录工具调用

3. **doom loop 检测**：在工具执行后检查 `checker.is_doom_loop`，如果为 True 则产出 ERROR 事件并终止

4. **软限制逻辑保持**：`check()` 返回 `warn` 时仍注入系统消息

具体修改位置：
- `ReactLoop.run()` 中的 LLM 调用段：根据 `checker.is_last_step` 决定是否传 tools
- `ReactLoop.run()` 中的工具执行段后：新增 doom loop 检测
- `ReactLoop.run()` 中的限制检查段：适配新的 `check()` 返回值（`max_steps` 作为 stop reason）

**Step 4: 运行全部循环体测试**

Run: `source venv/bin/activate && pytest tests/test_loop.py -v`
Expected: 全部 PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/loop.py tests/test_loop.py
git commit -m "refactor(loop): 双模式步数限制 + doom loop 检测"
```

---

### Task 5: 适配 agent.py 和 test_agent.py

**Files:**
- Modify: `src/pure_agent_loop/agent.py`
- Modify: `tests/test_agent.py`

**Step 1: 修改 test_agent.py 中的受影响测试**

需要修改的测试：
- `test_constructor_with_custom_limits`：去掉 `timeout=60.0` 参数
- `test_builtin_tools.py` 中 `test_invalid_status_raises_error`：`in_progress` 现在是合法状态，改测其他无效值
- `test_valid_statuses_constant`：更新为四状态

**Step 2: 运行测试确认失败**

Run: `source venv/bin/activate && pytest tests/test_agent.py -v`
Expected: 相关测试 FAIL

**Step 3: 修改 agent.py**

改动较小：
- `Agent.__init__` 中的 `self._limits = limits or LoopLimits()` 保持不变（LoopLimits 接口兼容）
- 无需其他改动，因为 Agent 只是将 LoopLimits 传递给 ReactLoop

**Step 4: 运行全部测试**

Run: `source venv/bin/activate && pytest tests/test_agent.py -v`
Expected: 全部 PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/agent.py tests/test_agent.py
git commit -m "refactor(agent): 适配新 LoopLimits 接口"
```

---

### Task 6: 更新 __init__.py 公共 API 导出

**Files:**
- Modify: `src/pure_agent_loop/__init__.py`

**Step 1: 检查是否需要新增导出**

检查 `LimitResult` 是否需要导出（当前未导出，保持不变即可）。`VALID_STATUSES` 和 `VALID_PRIORITIES` 如果作为公共常量可考虑导出。

**Step 2: 如有需要，更新导出列表**

可能需要新增 `VALID_PRIORITIES` 到 `__all__`，或保持不变（按最小变更原则）。

**Step 3: 提交（如有改动）**

```bash
git add src/pure_agent_loop/__init__.py
git commit -m "chore: 更新公共 API 导出"
```

---

### Task 7: 全量测试验证

**Step 1: 运行全部测试**

Run: `source venv/bin/activate && pytest --tb=short -v`
Expected: 全部 PASS

**Step 2: 运行覆盖率检查**

Run: `source venv/bin/activate && pytest --cov=pure_agent_loop --cov-report=term-missing`
Expected: 关键模块覆盖率 > 80%

**Step 3: 最终提交（如果有遗漏的修复）**

```bash
git add -A
git commit -m "test: 全量测试通过，重构完成"
```
