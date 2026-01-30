# 并行工具调用设计方案

## 概述

优化 `pure-agent-loop` 框架的工具执行机制，支持 LLM 在一次响应中返回多个独立的工具调用时并行执行，而非顺序等待。

## 背景

当前框架在 `loop.py:138-163` 中串行执行工具调用：

```python
for tc in response.tool_calls:
    result = await self._tools.execute(tc.name, tc.arguments)  # 逐个等待
```

即使 LLM 返回多个相互独立的 tool_calls（如多角度网络搜索），也必须依次执行，浪费时间。

## 设计决策

| 决策项 | 选择 | 理由 |
|--------|------|------|
| 并行策略 | LLM 自动判断 + 系统提示词引导 | 简单优雅，无需显式依赖声明 |
| 事件产出顺序 | 批量模式（先 ACTION 后 OBSERVATION） | 事件流清晰，便于 UI 展示 |
| 并发限制 | 无限制 | 信任 LLM 判断，符合 KISS 原则 |
| 错误处理 | 独立处理 | 与现有设计一致，失败不影响其他工具 |
| 执行计时 | 各工具独立计时 | 便于性能分析和慢工具识别 |

## 实现方案

### 1. `loop.py` 变更

使用 `asyncio.gather()` 实现并行执行：

```python
# ---- 新增辅助方法 ----
async def _execute_with_timing(
    self, name: str, arguments: dict[str, Any]
) -> tuple[str, float]:
    """执行单个工具并计时"""
    start_time = time.time()
    result = await self._tools.execute(name, arguments)
    duration = time.time() - start_time
    return result, duration

# ---- run() 方法中的工具执行逻辑 ----
if response.has_tool_calls:
    # Thought（如果有文本内容）
    if response.content:
        yield Event.thought(step=step, content=response.content)

    # 构建 assistant 消息（保持不变）
    assistant_msg = {...}
    msg_history.append(assistant_msg)

    # 1. 先产出所有 ACTION 事件
    for tc in response.tool_calls:
        yield Event.action(step=step, tool=tc.name, args=tc.arguments)

    # 2. 并行执行所有工具
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
        msg_history.append({
            "role": "tool",
            "tool_call_id": tc.id,
            "content": result,
        })
```

### 2. `prompts.py` 变更

在系统提示词中添加并行工具调用引导：

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
✅ 正确：一次返回两个 web_search 调用
❌ 错误：先搜索 Python，等待结果，再搜索 Rust
```

### 3. 测试用例

在 `tests/test_loop.py` 中新增：

#### 3.1 基础并行执行测试

```python
async def test_parallel_tool_execution_event_order():
    """验证并行执行时的事件产出顺序：先所有 ACTION，后所有 OBSERVATION"""
    # MockLLM 返回包含 3 个 tool_calls 的响应
    # 验证事件顺序：ACTION, ACTION, ACTION, OBSERVATION, OBSERVATION, OBSERVATION
```

#### 3.2 并行性能验证测试

```python
async def test_parallel_execution_performance():
    """验证并行执行确实节省时间"""
    # 3 个各耗时 0.1 秒的工具
    # 总耗时应 < 0.2 秒（而非串行的 0.3 秒）
```

#### 3.3 部分失败场景测试

```python
async def test_parallel_execution_partial_failure():
    """验证部分工具失败不影响其他工具"""
    # 3 个工具中 1 个抛异常
    # 验证其他 2 个正常返回，失败的返回错误信息
```

#### 3.4 混合场景测试

```python
async def test_mixed_parallel_and_sequential():
    """验证并行和串行调用的混合场景"""
    # 第一轮：3 个并行调用
    # 第二轮：1 个单独调用
    # 第三轮：最终回答
```

## 变更范围

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `src/pure_agent_loop/loop.py` | 修改 | 工具执行逻辑改为并行 |
| `src/pure_agent_loop/prompts.py` | 修改 | 添加并行调用引导 |
| `tests/test_loop.py` | 修改 | 新增 4 个测试用例 |

## 不变更

- `events.py` - 现有事件类型足够
- `tool.py` - 工具执行逻辑不变
- `agent.py` - 无需修改

## 向后兼容性

- ✅ 单工具调用行为完全不变
- ✅ 事件类型和数据结构不变
- ✅ API 无破坏性变更
- ✅ 现有测试无需修改

## 实施步骤

1. **修改 `loop.py`**：实现并行执行逻辑
2. **修改 `prompts.py`**：添加系统提示词引导
3. **编写测试**：新增 4 个测试用例
4. **运行测试**：确保所有测试通过
5. **更新文档**：在 CLAUDE.md 中说明并行执行特性
