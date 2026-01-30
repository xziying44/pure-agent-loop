# TodoWrite 简化设计：移除 in_progress 状态

## 背景

原有 TodoWrite 工具支持三种任务状态：`pending`、`in_progress`、`completed`。在实际使用中发现，每次 LLM 开始执行任务时标记 `in_progress` 会触发一次完整的 `todo_write` 调用，导致：

1. **上下文膨胀**：每次调用都向 LLM 上下文注入完整的 todo 列表副本
2. **Token 浪费**：`in_progress` 状态的语义价值有限，但消耗了额外的 API 调用
3. **行为冗余**：任务"正在执行"的状态可以从对话流程本身推断出来

## 设计决策

### 简化为二态模型

| 原状态 | 新状态 | 说明 |
|--------|--------|------|
| `pending` | `pending` | 保留，表示待完成 |
| `in_progress` | **删除** | 移除，不再支持 |
| `completed` | `completed` | 保留，表示已完成 |

### 无效状态处理：严格模式

当 LLM 传入 `in_progress` 或其他无效状态值时：

- `TodoItem.__post_init__()` 抛出 `ValueError`
- `TodoStore.write()` 捕获异常，返回错误字符串（不中断循环）
- 原有 todo 列表保持不变（部分失败不污染状态）

## 改动文件

### 源代码

| 文件 | 改动 |
|------|------|
| `src/pure_agent_loop/builtin_tools.py` | 新增 `VALID_STATUSES` 常量、`TodoItem.__post_init__()` 校验、`TodoStore.write()` 异常捕获、移除 `in_progress` 图标和统计、JSON Schema enum 二值化 |
| `src/pure_agent_loop/prompts.py` | 删除"标记 in_progress"指导、简化 todo 使用规范 |

### 测试代码

| 文件 | 改动 |
|------|------|
| `tests/test_builtin_tools.py` | 新增校验测试、移除 `in_progress` 用例 |
| `tests/test_agent.py` | 测试数据 `in_progress` → `pending` |
| `tests/test_loop.py` | 测试数据 `in_progress` → `completed` |
| `tests/test_events.py` | 测试数据 `in_progress` → `pending` |
| `tests/test_renderer.py` | 测试数据 `in_progress` → `pending` |

## API 变更

### TodoItem

```python
# 新增常量
VALID_STATUSES = ("pending", "completed")

@dataclass
class TodoItem:
    content: str
    status: str = "pending"

    def __post_init__(self):
        if self.status not in VALID_STATUSES:
            raise ValueError(f"无效的任务状态: '{self.status}'，仅支持: {', '.join(VALID_STATUSES)}")
```

### TodoStore.write()

```python
def write(self, todos: list[dict[str, str]]) -> str:
    try:
        self._todos = [TodoItem(**t) for t in todos]
    except ValueError as e:
        return f"❌ 任务更新失败: {e}"
    return self._format_output()
```

### JSON Schema

```python
"status": {
    "type": "string",
    "enum": ["pending", "completed"],  # 原为 ["pending", "in_progress", "completed"]
    "description": "任务状态",
}
```

## 向后兼容性

- **框架内部**：无兼容问题，状态值仅在单次 Agent 运行期间存在
- **用户代码**：如果用户代码硬编码了 `in_progress` 检查，需要移除
- **提示词**：已更新内置提示词，用户自定义提示词如有 `in_progress` 引用需自行调整

## 测试验证

```bash
pytest --tb=short
# 122 passed
```
