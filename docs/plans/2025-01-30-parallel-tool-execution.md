# 并行工具调用实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 支持 LLM 返回多个独立 tool_calls 时并行执行，提升任务处理效率

**Architecture:** 使用 `asyncio.gather()` 并行执行工具，采用批量事件模式（先产出所有 ACTION，再产出所有 OBSERVATION），并通过系统提示词引导 LLM 主动返回可并行的多工具调用

**Tech Stack:** Python asyncio, pytest, pytest-asyncio

---

## Task 1: 新增多工具调用的 Mock 响应工厂函数

**Files:**
- Modify: `tests/test_loop.py:39-46`

**Step 1: 添加多工具调用响应工厂函数**

在 `_tool_call_response` 函数后添加新函数：

```python
def _multi_tool_call_response(calls: list[tuple[str, dict]]) -> LLMResponse:
    """创建多工具调用响应

    Args:
        calls: [(tool_name, arguments), ...] 格式的调用列表
    """
    return LLMResponse(
        content=None,
        tool_calls=[
            ToolCall(id=f"call_{name}_{i}", name=name, arguments=args)
            for i, (name, args) in enumerate(calls)
        ],
        usage=TokenUsage(prompt_tokens=50, completion_tokens=30, total_tokens=80),
        raw={},
    )
```

**Step 2: 验证语法正确**

Run: `python -c "from tests.test_loop import _multi_tool_call_response; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add tests/test_loop.py
git commit -m "test: 添加多工具调用响应工厂函数"
```

---

## Task 2: 编写并行执行事件顺序测试（失败测试）

**Files:**
- Modify: `tests/test_loop.py`

**Step 1: 编写测试 - 验证事件顺序**

在 `TestReactLoop` 类末尾添加：

```python
@pytest.mark.asyncio
async def test_parallel_tool_execution_event_order(self):
    """并行执行时应先产出所有 ACTION，再产出所有 OBSERVATION"""
    import asyncio

    @tool
    async def slow_tool(name: str) -> str:
        """模拟耗时工具"""
        await asyncio.sleep(0.05)
        return f"result_{name}"

    registry = ToolRegistry()
    registry.register(slow_tool)

    llm = MockLLM([
        _multi_tool_call_response([
            ("slow_tool", {"name": "a"}),
            ("slow_tool", {"name": "b"}),
            ("slow_tool", {"name": "c"}),
        ]),
        _text_response("完成"),
    ])

    loop = ReactLoop(
        llm=llm,
        tool_registry=registry,
        limits=LoopLimits(),
        retry=RetryConfig(),
    )

    events = []
    async for event in loop.run("测试并行"):
        events.append(event)

    # 提取 ACTION 和 OBSERVATION 事件的索引
    action_indices = [i for i, e in enumerate(events) if e.type == EventType.ACTION]
    obs_indices = [i for i, e in enumerate(events) if e.type == EventType.OBSERVATION]

    # 应该有 3 个 ACTION 和 3 个 OBSERVATION
    assert len(action_indices) == 3, f"期望 3 个 ACTION，实际 {len(action_indices)}"
    assert len(obs_indices) == 3, f"期望 3 个 OBSERVATION，实际 {len(obs_indices)}"

    # 所有 ACTION 应该在所有 OBSERVATION 之前
    assert max(action_indices) < min(obs_indices), \
        "所有 ACTION 事件应该在所有 OBSERVATION 事件之前"
```

**Step 2: 运行测试验证失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_loop.py::TestReactLoop::test_parallel_tool_execution_event_order -v`

Expected: FAIL（当前串行实现会交错产出 ACTION 和 OBSERVATION）

**Step 3: Commit 失败测试**

```bash
git add tests/test_loop.py
git commit -m "test: 添加并行执行事件顺序测试（红）"
```

---

## Task 3: 编写并行性能验证测试（失败测试）

**Files:**
- Modify: `tests/test_loop.py`

**Step 1: 编写测试 - 验证并行确实节省时间**

```python
@pytest.mark.asyncio
async def test_parallel_execution_performance(self):
    """并行执行应该节省时间：3 个 0.1 秒工具应在 0.2 秒内完成"""
    import asyncio
    import time

    @tool
    async def timed_tool(name: str) -> str:
        """耗时 0.1 秒的工具"""
        await asyncio.sleep(0.1)
        return f"done_{name}"

    registry = ToolRegistry()
    registry.register(timed_tool)

    llm = MockLLM([
        _multi_tool_call_response([
            ("timed_tool", {"name": "a"}),
            ("timed_tool", {"name": "b"}),
            ("timed_tool", {"name": "c"}),
        ]),
        _text_response("完成"),
    ])

    loop = ReactLoop(
        llm=llm,
        tool_registry=registry,
        limits=LoopLimits(),
        retry=RetryConfig(),
    )

    start = time.time()
    events = []
    async for event in loop.run("测试性能"):
        events.append(event)
    elapsed = time.time() - start

    # 串行需要 0.3 秒，并行应该 < 0.2 秒
    assert elapsed < 0.2, f"并行执行耗时 {elapsed:.3f}s，应该 < 0.2s"
```

**Step 2: 运行测试验证失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_loop.py::TestReactLoop::test_parallel_execution_performance -v`

Expected: FAIL（当前串行实现需要约 0.3 秒）

**Step 3: Commit**

```bash
git add tests/test_loop.py
git commit -m "test: 添加并行执行性能测试（红）"
```

---

## Task 4: 编写部分失败场景测试（失败测试）

**Files:**
- Modify: `tests/test_loop.py`

**Step 1: 编写测试 - 部分工具失败不影响其他**

```python
@pytest.mark.asyncio
async def test_parallel_execution_partial_failure(self):
    """部分工具失败不应影响其他工具的执行"""

    @tool
    def success_tool(x: str) -> str:
        """成功的工具"""
        return f"success_{x}"

    @tool
    def fail_tool(x: str) -> str:
        """失败的工具"""
        raise ValueError("故意失败")

    registry = ToolRegistry()
    registry.register(success_tool)
    registry.register(fail_tool)

    llm = MockLLM([
        _multi_tool_call_response([
            ("success_tool", {"x": "a"}),
            ("fail_tool", {"x": "b"}),
            ("success_tool", {"x": "c"}),
        ]),
        _text_response("处理完成"),
    ])

    loop = ReactLoop(
        llm=llm,
        tool_registry=registry,
        limits=LoopLimits(),
        retry=RetryConfig(),
    )

    events = []
    async for event in loop.run("测试部分失败"):
        events.append(event)

    # 应该有 3 个 OBSERVATION
    obs_events = [e for e in events if e.type == EventType.OBSERVATION]
    assert len(obs_events) == 3, f"期望 3 个 OBSERVATION，实际 {len(obs_events)}"

    # 检查结果：2 个成功，1 个包含错误信息
    results = [e.data["result"] for e in obs_events]
    success_count = sum(1 for r in results if r.startswith("success_"))
    error_count = sum(1 for r in results if "执行失败" in r)

    assert success_count == 2, f"期望 2 个成功，实际 {success_count}"
    assert error_count == 1, f"期望 1 个错误，实际 {error_count}"

    # 循环应该正常结束
    end_event = next(e for e in events if e.type == EventType.LOOP_END)
    assert end_event.data["stop_reason"] == "completed"
```

**Step 2: 运行测试验证通过（当前实现已支持）**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_loop.py::TestReactLoop::test_parallel_execution_partial_failure -v`

Expected: PASS（当前实现的 `Tool.execute()` 已捕获异常）

**Step 3: Commit**

```bash
git add tests/test_loop.py
git commit -m "test: 添加并行执行部分失败测试"
```

---

## Task 5: 编写各工具独立计时测试（失败测试）

**Files:**
- Modify: `tests/test_loop.py`

**Step 1: 编写测试 - 验证每个工具有独立 duration**

```python
@pytest.mark.asyncio
async def test_parallel_execution_individual_timing(self):
    """每个工具应该有独立的执行时间记录"""
    import asyncio

    @tool
    async def fast_tool(x: str) -> str:
        """快速工具 0.05 秒"""
        await asyncio.sleep(0.05)
        return "fast"

    @tool
    async def slow_tool(x: str) -> str:
        """慢速工具 0.15 秒"""
        await asyncio.sleep(0.15)
        return "slow"

    registry = ToolRegistry()
    registry.register(fast_tool)
    registry.register(slow_tool)

    llm = MockLLM([
        _multi_tool_call_response([
            ("fast_tool", {"x": "a"}),
            ("slow_tool", {"x": "b"}),
        ]),
        _text_response("完成"),
    ])

    loop = ReactLoop(
        llm=llm,
        tool_registry=registry,
        limits=LoopLimits(),
        retry=RetryConfig(),
    )

    events = []
    async for event in loop.run("测试计时"):
        events.append(event)

    obs_events = [e for e in events if e.type == EventType.OBSERVATION]
    assert len(obs_events) == 2

    # 获取每个工具的 duration
    durations = {e.data["tool"]: e.data["duration"] for e in obs_events}

    # fast_tool 应该约 0.05s，slow_tool 应该约 0.15s
    assert durations["fast_tool"] < 0.1, f"fast_tool duration {durations['fast_tool']:.3f}s 应该 < 0.1s"
    assert durations["slow_tool"] > 0.1, f"slow_tool duration {durations['slow_tool']:.3f}s 应该 > 0.1s"
    # 两者差距应该明显
    assert durations["slow_tool"] > durations["fast_tool"] * 2, \
        "slow_tool 耗时应该明显大于 fast_tool"
```

**Step 2: 运行测试验证失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_loop.py::TestReactLoop::test_parallel_execution_individual_timing -v`

Expected: FAIL（当前串行实现中 duration 不独立）

**Step 3: Commit**

```bash
git add tests/test_loop.py
git commit -m "test: 添加并行执行独立计时测试（红）"
```

---

## Task 6: 实现 `_execute_with_timing` 辅助方法

**Files:**
- Modify: `src/pure_agent_loop/loop.py:49` （在 `__init__` 后添加新方法）

**Step 1: 添加辅助方法**

在 `ReactLoop` 类的 `__init__` 方法后（约第 49 行）添加：

```python
async def _execute_with_timing(
    self, name: str, arguments: dict[str, Any]
) -> tuple[str, float]:
    """执行单个工具并计时

    Args:
        name: 工具名称
        arguments: 工具参数

    Returns:
        (执行结果, 耗时秒数)
    """
    start_time = time.time()
    result = await self._tools.execute(name, arguments)
    duration = time.time() - start_time
    return result, duration
```

**Step 2: 验证语法正确**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && python -c "from pure_agent_loop.loop import ReactLoop; print('OK')"`

Expected: `OK`

**Step 3: Commit**

```bash
git add src/pure_agent_loop/loop.py
git commit -m "feat: 添加工具执行计时辅助方法"
```

---

## Task 7: 实现并行工具执行逻辑

**Files:**
- Modify: `src/pure_agent_loop/loop.py:137-163`

**Step 1: 导入 asyncio（如果尚未导入）**

确认文件头部有 `import asyncio`（如无则添加）。

**Step 2: 替换工具执行逻辑**

将第 137-163 行的串行执行代码替换为：

```python
                # ---- 执行工具（并行） ----
                # 1. 先产出所有 ACTION 事件
                for tc in response.tool_calls:
                    yield Event.action(step=step, tool=tc.name, args=tc.arguments)

                # 2. 并行执行所有工具
                import asyncio
                results_with_timing = await asyncio.gather(*[
                    self._execute_with_timing(tc.name, tc.arguments)
                    for tc in response.tool_calls
                ])

                # 3. 批量产出 OBSERVATION 事件并更新消息历史
                for tc, (result, duration) in zip(response.tool_calls, results_with_timing):
                    yield Event.observation(
                        step=step, tool=tc.name, result=result, duration=duration
                    )

                    # 如果是 todo_write 工具，额外产出 TODO_UPDATE 事件
                    if tc.name == "todo_write" and self._todo_store is not None:
                        yield Event.todo_update(
                            step=step,
                            todos=[t.to_dict() for t in self._todo_store.todos],
                        )

                    # 将工具结果追加到消息历史
                    msg_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )
```

**Step 3: 运行所有并行测试验证通过**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_loop.py -k "parallel" -v`

Expected: 4 个并行测试全部 PASS

**Step 4: 运行完整测试套件确保无回归**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_loop.py -v`

Expected: 全部 PASS

**Step 5: Commit**

```bash
git add src/pure_agent_loop/loop.py
git commit -m "feat: 实现工具并行执行"
```

---

## Task 8: 更新系统提示词 - 添加并行调用引导

**Files:**
- Modify: `src/pure_agent_loop/prompts.py:36` （在"先思考，再行动"段落后添加）

**Step 1: 添加并行工具使用引导**

在 `## 先思考，再行动` 段落结束后（约第 43 行 `绝对禁止:` 之前）添加：

```python
## 高效工具使用
当你判断有多个**相互独立**的工具调用可以同时进行时，应该在一次回复中返回多个工具调用，而不是逐个调用等待结果。

### 适合并行调用的场景：
- 多角度搜索：从不同关键词或来源搜索信息
- 批量查询：同时查询多个文件、API 或数据库
- 独立验证：对同一问题从多个途径获取佐证

### 不适合并行调用的场景：
- 后续调用依赖前一个结果（如：先搜索文件路径，再读取文件内容）
- 需要根据前一个结果决定下一步策略

示例：用户要求"搜索 Python 和 Rust 的异步编程最佳实践"
✅ 正确：一次返回两个搜索工具调用
❌ 错误：先搜索 Python，等待结果，再搜索 Rust

```

**Step 2: 验证提示词生成正确**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && python -c "from pure_agent_loop.prompts import build_system_prompt; p = build_system_prompt(); assert '并行调用' in p; print('OK')"`

Expected: `OK`

**Step 3: 运行提示词测试**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_prompts.py -v`

Expected: PASS

**Step 4: Commit**

```bash
git add src/pure_agent_loop/prompts.py
git commit -m "feat: 系统提示词添加并行工具调用引导"
```

---

## Task 9: 添加并行工具调用引导的测试

**Files:**
- Modify: `tests/test_prompts.py`

**Step 1: 添加测试用例**

```python
def test_system_prompt_includes_parallel_tool_guidance():
    """系统提示词应包含并行工具调用引导"""
    prompt = build_system_prompt()

    # 检查关键内容
    assert "高效工具使用" in prompt
    assert "并行调用" in prompt
    assert "多角度搜索" in prompt
    assert "不适合并行调用" in prompt
```

**Step 2: 运行测试**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_prompts.py::test_system_prompt_includes_parallel_tool_guidance -v`

Expected: PASS

**Step 3: Commit**

```bash
git add tests/test_prompts.py
git commit -m "test: 添加并行工具调用引导测试"
```

---

## Task 10: 运行完整测试套件并验收

**Files:**
- None (验证)

**Step 1: 运行全部测试**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest --cov=pure_agent_loop --cov-report=term-missing`

Expected: 全部 PASS，覆盖率应保持或提升

**Step 2: 检查 git 状态**

Run: `git status`

Expected: 工作区干净

**Step 3: 查看提交历史**

Run: `git log --oneline -10`

Expected: 看到本次实施的所有提交

---

## 验收清单

- [ ] 4 个并行执行测试全部通过
- [ ] 所有现有测试无回归
- [ ] 系统提示词包含并行调用引导
- [ ] 代码覆盖率保持或提升
- [ ] 每个任务都有独立提交
