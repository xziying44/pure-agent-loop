# Conversation 多轮会话设计

## 背景

当前 `Agent.arun_stream(task)` 每次调用都是干净的上下文——`ReactLoop` 在没有 `messages` 参数时会新建消息历史。虽然 `arun_stream(task, messages=...)` 已支持传入历史消息实现续接，但在流式场景下需要手动从 LOOP_END 事件中提取消息，使用不够方便。

## 目标

提供一个独立的 `Conversation` 对象，自动维护消息历史，让多轮对话续接变得自然直观。

## 设计

### 核心思路

`Conversation` 是 Agent 的薄包装器：
- 内部维护 `list[dict]` 消息历史
- 每次 `send` 时把历史传给 `agent.arun_stream(task, messages=history)`
- 从 LOOP_END 事件中提取更新后的消息列表
- Agent 保持无状态，Conversation 管理状态

### Conversation 类

```python
class Conversation:
    """多轮对话会话

    由 Agent.conversation() 创建，自动维护消息历史。
    同一 Agent 可创建多个独立 Conversation。
    """

    def __init__(self, agent: Agent):
        self._agent = agent
        self._messages: list[dict[str, Any]] = []

    @property
    def messages(self) -> list[dict[str, Any]]:
        """当前消息历史（只读副本）"""
        return list(self._messages)

    def reset(self) -> None:
        """清空消息历史，开始新对话"""
        self._messages = []

    async def send_stream(self, task: str) -> AsyncIterator[Event]:
        """发送消息并流式返回事件"""
        msgs = self._messages if self._messages else None
        async for event in self._agent.arun_stream(task, messages=msgs):
            if event.type == EventType.LOOP_END:
                self._messages = event.data.get("messages", [])
            yield event

    async def send(self, task: str) -> AgentResult:
        """发送消息并返回完整结果"""
        events: list[Event] = []
        async for event in self.send_stream(task):
            events.append(event)
        return self._agent._build_result(events)

    def send_stream_sync(self, task: str) -> Iterator[Event]:
        """同步流式发送"""
        ...

    def send_sync(self, task: str) -> AgentResult:
        """同步发送"""
        ...
```

### Agent 侧变更

仅新增一个工厂方法：

```python
class Agent:
    def conversation(self) -> Conversation:
        """创建一个新的多轮对话会话"""
        return Conversation(self)
```

### 消息历史流转

```
conv.send_stream("任务1")
  → agent.arun_stream("任务1", messages=None)    # 新对话
  → LOOP_END 带回 messages=[sys, user1, asst1, ...]
  → self._messages 更新

conv.send_stream("任务2")
  → agent.arun_stream("任务2", messages=[sys, user1, asst1, ...])  # 续接
  → LOOP_END 带回 messages=[..., user2, asst2, ...]
  → self._messages 更新
```

### 使用示例

```python
agent = Agent(name="助手", model="deepseek-chat", ...)
conv = agent.conversation()

# 多轮对话自动续接
async for event in conv.send_stream("设计一个聊天页面"):
    ...
async for event in conv.send_stream("在基础上新增日志管理"):
    ...

# 全新对话
conv2 = agent.conversation()
async for event in conv2.send_stream("写一个完全不同的任务"):
    ...
```

## 文件变更范围

| 文件 | 变更类型 | 说明 |
|------|----------|------|
| `src/pure_agent_loop/conversation.py` | 新建 | Conversation 类 |
| `src/pure_agent_loop/agent.py` | 修改 | 新增 `conversation()` 工厂方法 |
| `src/pure_agent_loop/__init__.py` | 修改 | 导出 `Conversation` |
| `tests/test_conversation.py` | 新建 | 测试 |
| `examples/conversation.py` | 新建 | 使用示例 |

**零修改**：`loop.py`、`events.py`、`limits.py` 等核心模块。

## 设计决策

- **Agent 保持无状态**：状态仅存在于 Conversation 对象中
- **复用现有 messages 机制**：不修改 ReactLoop，仅在上层包装
- **LOOP_END 事件驱动**：通过已有事件提取消息历史，无需新增接口
- **浅拷贝安全**：ReactLoop.run() 在接收 messages 时做 `list(messages)` 浅拷贝
