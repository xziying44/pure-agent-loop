# 内置系统提示词 & TodoWrite 工具 — 实施计划

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** 为 pure-agent-loop 框架新增内置系统提示词模板（模板包裹模式 + 智能体名称注入）和 TodoWrite 内置任务管理工具（自动注册 + TODO_UPDATE 事件流）。

**Architecture:** Agent 初始化时自动创建 TodoStore 和 todo_write 工具，注册到工具列表；通过 `build_system_prompt()` 将用户 system_prompt 注入内置模板；ReactLoop 在执行 todo_write 后额外产出 TODO_UPDATE 事件；AgentResult 新增 todos 属性暴露最终状态。

**Tech Stack:** Python 3.10+, dataclasses, pytest, pytest-asyncio

---

## Task 1: 新增 `prompts.py` — 内置系统提示词模板

**Files:**
- Create: `src/pure_agent_loop/prompts.py`
- Test: `tests/test_prompts.py`

**Step 1: 编写测试 `tests/test_prompts.py`**

```python
"""内置系统提示词模板测试"""

import pytest
from pure_agent_loop.prompts import build_system_prompt


class TestBuildSystemPrompt:
    """build_system_prompt 测试"""

    def test_default_name(self):
        """默认名称应为 '智能助理'"""
        prompt = build_system_prompt()
        assert "你是智能助理" in prompt

    def test_custom_name(self):
        """自定义名称应注入到角色描述"""
        prompt = build_system_prompt(name="研究助手")
        assert "你是研究助手" in prompt

    def test_user_prompt_injected(self):
        """用户自定义提示词应被注入"""
        prompt = build_system_prompt(user_prompt="你擅长数学计算。")
        assert "你擅长数学计算。" in prompt

    def test_empty_user_prompt(self):
        """空用户提示词不应导致异常"""
        prompt = build_system_prompt(user_prompt="")
        assert "你是智能助理" in prompt

    def test_contains_todo_requirement(self):
        """提示词应包含 TodoWrite 使用要求"""
        prompt = build_system_prompt()
        assert "todo_write" in prompt

    def test_contains_react_guidance(self):
        """提示词应包含思考-行动指导"""
        prompt = build_system_prompt()
        assert "思考" in prompt
        assert "行动" in prompt

    def test_contains_role_section(self):
        """提示词应包含角色描述段"""
        prompt = build_system_prompt()
        assert "# Role" in prompt or "# 角色" in prompt
```

**Step 2: 运行测试确认失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_prompts.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pure_agent_loop.prompts'`

**Step 3: 实现 `src/pure_agent_loop/prompts.py`**

```python
"""内置系统提示词模板

提供框架级的系统提示词构建，将用户自定义提示词注入到结构化模板中。
"""


def build_system_prompt(
    name: str = "智能助理",
    user_prompt: str = "",
) -> str:
    """构建完整的系统提示词

    将智能体名称和用户自定义提示词注入到内置模板中。

    Args:
        name: 智能体名称，注入到角色描述中
        user_prompt: 用户自定义的任务指令

    Returns:
        完整的系统提示词字符串
    """
    user_section = ""
    if user_prompt.strip():
        user_section = f"""
# 用户自定义指令
{user_prompt.strip()}
"""

    return f"""# Role
你是{name}，一个运行在事件循环中的高级自治智能体。你的核心职责是通过逻辑推理（Reasoning）和行动（Acting）的循环来解决复杂问题。

# 核心行为准则

## 先思考，再行动
在每一步中，你必须：
1. **思考**：在回复内容中清晰阐述你的分析、推理和计划。这些思考内容会被保存到对话历史中，帮助你在后续步骤中保持上下文连贯，避免重复工作或遗忘关键信息。
2. **行动**：基于思考结论，选择合适的工具执行。不要在没有充分思考的情况下直接调用工具。

绝对禁止：
- 跳过思考直接调用工具
- 编造工具未返回的信息（禁止幻觉）
- 忽略工具返回的错误，必须在思考中反思并调整策略

## 任务管理（极其重要）
你**必须**使用 todo_write 工具来管理和规划任务。这是强制要求，不可忽略。

### 何时使用 todo_write：
- **收到任务后立即使用**：将任务拆解为具体的子步骤
- **开始某个子任务前**：将其状态标记为 in_progress
- **完成某个子任务后立即**：将其状态标记为 completed
- **发现新的子任务时**：追加到列表中

### todo_write 使用规范：
- 每个 todo 项必须包含 content（任务内容）和 status（pending/in_progress/completed）
- 同一时刻只能有一个 todo 处于 in_progress 状态
- 完成后必须立即标记，不要批量标记
- 复杂任务必须拆解为 3 个以上的子步骤

# 约束条件
1. 如果工具返回错误或无结果，在思考中诚实反思，尝试调整策略，不要编造结果
2. 时刻关注工具返回的新信息，基于事实推进任务
3. 任务完成时，在最终回复中总结成果
{user_section}""".strip()
```

**Step 4: 运行测试确认通过**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_prompts.py -v`
Expected: 全部 PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/prompts.py tests/test_prompts.py
git commit -m "feat: 添加内置系统提示词模板 (prompts.py)"
```

---

## Task 2: 新增 `builtin_tools.py` — TodoStore + todo_write 工具

**Files:**
- Create: `src/pure_agent_loop/builtin_tools.py`
- Test: `tests/test_builtin_tools.py`

**Step 1: 编写测试 `tests/test_builtin_tools.py`**

```python
"""内置工具测试"""

import pytest
from pure_agent_loop.builtin_tools import TodoItem, TodoStore, create_todo_tool
from pure_agent_loop.tool import Tool


class TestTodoItem:
    """TodoItem 数据类测试"""

    def test_create_default(self):
        """默认状态应为 pending"""
        item = TodoItem(content="测试任务")
        assert item.content == "测试任务"
        assert item.status == "pending"

    def test_create_with_status(self):
        """应支持指定状态"""
        item = TodoItem(content="进行中", status="in_progress")
        assert item.status == "in_progress"

    def test_to_dict(self):
        """应能转换为字典"""
        item = TodoItem(content="任务A", status="completed")
        d = item.to_dict()
        assert d == {"content": "任务A", "status": "completed"}


class TestTodoStore:
    """TodoStore 测试"""

    def test_initial_empty(self):
        """初始应为空列表"""
        store = TodoStore()
        assert store.todos == []

    def test_write_replaces_list(self):
        """write 应完全替换 todo 列表"""
        store = TodoStore()
        store.write([
            {"content": "任务1", "status": "pending"},
            {"content": "任务2", "status": "in_progress"},
        ])
        assert len(store.todos) == 2
        assert store.todos[0].content == "任务1"
        assert store.todos[1].status == "in_progress"

    def test_write_returns_formatted_string(self):
        """write 返回值应包含格式化的 todo 列表"""
        store = TodoStore()
        result = store.write([
            {"content": "搜索资料", "status": "completed"},
            {"content": "分析结果", "status": "in_progress"},
            {"content": "撰写报告", "status": "pending"},
        ])
        assert "搜索资料" in result
        assert "分析结果" in result
        assert "撰写报告" in result
        assert "completed" in result or "✅" in result

    def test_write_empty_list(self):
        """写入空列表应清空"""
        store = TodoStore()
        store.write([{"content": "任务", "status": "pending"}])
        result = store.write([])
        assert store.todos == []
        assert "空" in result

    def test_multiple_writes_replace(self):
        """多次 write 应完全替换"""
        store = TodoStore()
        store.write([{"content": "A", "status": "pending"}])
        store.write([{"content": "B", "status": "completed"}])
        assert len(store.todos) == 1
        assert store.todos[0].content == "B"

    def test_todos_property_returns_copy(self):
        """todos 属性应返回副本，不影响内部状态"""
        store = TodoStore()
        store.write([{"content": "A", "status": "pending"}])
        external = store.todos
        external.clear()
        assert len(store.todos) == 1


class TestCreateTodoTool:
    """create_todo_tool 工厂函数测试"""

    def test_creates_tool_instance(self):
        """应返回 Tool 实例"""
        store = TodoStore()
        t = create_todo_tool(store)
        assert isinstance(t, Tool)
        assert t.name == "todo_write"

    def test_tool_has_correct_schema(self):
        """工具 schema 应正确定义"""
        store = TodoStore()
        t = create_todo_tool(store)
        schema = t.to_openai_schema()
        assert schema["type"] == "function"
        assert schema["function"]["name"] == "todo_write"
        params = schema["function"]["parameters"]
        assert "todos" in params["properties"]

    async def test_tool_execute_updates_store(self):
        """执行工具应更新 store"""
        store = TodoStore()
        t = create_todo_tool(store)
        result = await t.execute({
            "todos": [
                {"content": "任务1", "status": "pending"},
                {"content": "任务2", "status": "in_progress"},
            ]
        })
        assert len(store.todos) == 2
        assert "任务1" in result
        assert "任务2" in result
```

**Step 2: 运行测试确认失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_builtin_tools.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'pure_agent_loop.builtin_tools'`

**Step 3: 实现 `src/pure_agent_loop/builtin_tools.py`**

```python
"""内置工具

框架自带的工具实现，包括任务管理工具 todo_write。
"""

import json
from dataclasses import dataclass, field
from typing import Any

from .tool import Tool


@dataclass
class TodoItem:
    """单个任务项

    Attributes:
        content: 任务内容描述
        status: 任务状态 (pending/in_progress/completed)
    """

    content: str
    status: str = "pending"

    def to_dict(self) -> dict[str, str]:
        """转换为字典"""
        return {"content": self.content, "status": self.status}


class TodoStore:
    """任务列表内存存储

    管理 Agent 运行期间的 todo 状态。每次 write() 调用完全替换列表。
    """

    def __init__(self):
        self._todos: list[TodoItem] = []

    def write(self, todos: list[dict[str, str]]) -> str:
        """替换整个 todo 列表

        Args:
            todos: 新的任务列表，每项包含 content 和 status

        Returns:
            格式化的当前 todo 列表字符串（注入 LLM 上下文）
        """
        self._todos = [TodoItem(**t) for t in todos]
        return self._format_output()

    @property
    def todos(self) -> list[TodoItem]:
        """获取当前 todo 列表（返回副本）"""
        return list(self._todos)

    def _format_output(self) -> str:
        """格式化当前 todo 列表"""
        if not self._todos:
            return "📋 任务列表为空"

        status_icons = {
            "pending": "⬜",
            "in_progress": "🔄",
            "completed": "✅",
        }

        lines = ["📋 当前任务列表："]
        for i, todo in enumerate(self._todos, 1):
            icon = status_icons.get(todo.status, "❓")
            lines.append(f"  {i}. {icon} [{todo.status}] {todo.content}")

        pending = sum(1 for t in self._todos if t.status == "pending")
        in_progress = sum(1 for t in self._todos if t.status == "in_progress")
        completed = sum(1 for t in self._todos if t.status == "completed")
        lines.append(
            f"\n总计: {len(self._todos)} 项 | "
            f"待处理: {pending} | 进行中: {in_progress} | 已完成: {completed}"
        )
        return "\n".join(lines)


def create_todo_tool(store: TodoStore) -> Tool:
    """创建绑定到指定 TodoStore 的 todo_write 工具

    Args:
        store: TodoStore 实例，工具执行时操作此 store

    Returns:
        Tool 实例
    """

    def todo_write(todos: list[dict[str, str]]) -> str:
        """更新任务列表，完全替换当前列表

        Args:
            todos: 任务列表，每项包含 content（任务内容）和 status（pending/in_progress/completed）
        """
        return store.write(todos)

    return Tool(
        name="todo_write",
        description="更新任务列表，完全替换当前列表。每个任务项包含 content（任务内容）和 status（pending/in_progress/completed）。",
        parameters={
            "type": "object",
            "properties": {
                "todos": {
                    "type": "array",
                    "description": "任务列表",
                    "items": {
                        "type": "object",
                        "properties": {
                            "content": {
                                "type": "string",
                                "description": "任务内容",
                            },
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"],
                                "description": "任务状态",
                            },
                        },
                        "required": ["content", "status"],
                    },
                },
            },
            "required": ["todos"],
        },
        function=todo_write,
        is_async=False,
    )
```

**Step 4: 运行测试确认通过**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_builtin_tools.py -v`
Expected: 全部 PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/builtin_tools.py tests/test_builtin_tools.py
git commit -m "feat: 添加 TodoStore 和 todo_write 内置工具 (builtin_tools.py)"
```

---

## Task 3: 修改 `events.py` — 新增 TODO_UPDATE 事件类型

**Files:**
- Modify: `src/pure_agent_loop/events.py:12-21` (EventType 枚举)
- Modify: `src/pure_agent_loop/events.py:97-104` (新增工厂方法)
- Test: `tests/test_events.py`

**Step 1: 补充测试到 `tests/test_events.py`**

在文件末尾追加：

```python

    def test_todo_update_event(self):
        """应该有 todo_update 工厂方法"""
        event = Event.todo_update(
            step=2,
            todos=[
                {"content": "任务A", "status": "completed"},
                {"content": "任务B", "status": "in_progress"},
            ],
        )
        assert event.type == EventType.TODO_UPDATE
        assert event.step == 2
        assert len(event.data["todos"]) == 2
        assert event.data["todos"][0]["content"] == "任务A"


class TestEventTypeTodoUpdate:
    """TODO_UPDATE 事件类型测试"""

    def test_todo_update_type_exists(self):
        """应该存在 TODO_UPDATE 事件类型"""
        assert EventType.TODO_UPDATE.value == "todo_update"
```

**Step 2: 运行测试确认失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_events.py -v`
Expected: FAIL — `AttributeError: 'EventType' object has no attribute 'TODO_UPDATE'`

**Step 3: 修改 `src/pure_agent_loop/events.py`**

在 EventType 枚举中 `LOOP_END` 后添加：
```python
    TODO_UPDATE = "todo_update"
```

在 Event 类末尾（`soft_limit` 方法之后）添加工厂方法：
```python
    @classmethod
    def todo_update(cls, step: int, todos: list[dict]) -> "Event":
        """创建任务列表变更事件"""
        return cls(
            type=EventType.TODO_UPDATE,
            step=step,
            data={"todos": todos},
        )
```

**Step 4: 运行测试确认通过**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_events.py -v`
Expected: 全部 PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/events.py tests/test_events.py
git commit -m "feat: 新增 TODO_UPDATE 事件类型"
```

---

## Task 4: 修改 `loop.py` — 支持 TODO_UPDATE 事件产出

**Files:**
- Modify: `src/pure_agent_loop/loop.py:33-45` (构造函数新增 todo_store)
- Modify: `src/pure_agent_loop/loop.py:134-153` (工具执行后检查并产出 TODO_UPDATE)
- Test: `tests/test_loop.py`

**Step 1: 补充测试到 `tests/test_loop.py`**

在文件末尾追加：

```python

    @pytest.mark.asyncio
    async def test_todo_write_emits_todo_update_event(self):
        """执行 todo_write 工具应额外产出 TODO_UPDATE 事件"""
        from pure_agent_loop.builtin_tools import TodoStore, create_todo_tool

        store = TodoStore()
        todo_tool = create_todo_tool(store)

        registry = ToolRegistry()
        registry.register(todo_tool)

        llm = MockLLM([
            _tool_call_response("todo_write", {
                "todos": [
                    {"content": "搜索资料", "status": "in_progress"},
                    {"content": "分析结果", "status": "pending"},
                ]
            }),
            _text_response("已规划任务"),
        ])

        loop = ReactLoop(
            llm=llm,
            tool_registry=registry,
            limits=LoopLimits(),
            retry=RetryConfig(),
            todo_store=store,
        )

        events = []
        async for event in loop.run("测试任务"):
            events.append(event)

        types = [e.type for e in events]
        assert EventType.TODO_UPDATE in types

        todo_event = next(e for e in events if e.type == EventType.TODO_UPDATE)
        assert len(todo_event.data["todos"]) == 2
        assert todo_event.data["todos"][0]["content"] == "搜索资料"

    @pytest.mark.asyncio
    async def test_no_todo_update_without_store(self):
        """未传入 todo_store 时不应产出 TODO_UPDATE 事件"""

        @tool
        def dummy(x: str) -> str:
            """空操作"""
            return "ok"

        registry = ToolRegistry()
        registry.register(dummy)

        llm = MockLLM([
            _tool_call_response("dummy", {"x": "test"}),
            _text_response("完成"),
        ])

        loop = ReactLoop(
            llm=llm,
            tool_registry=registry,
            limits=LoopLimits(),
            retry=RetryConfig(),
        )

        events = []
        async for event in loop.run("测试"):
            events.append(event)

        types = [e.type for e in events]
        assert EventType.TODO_UPDATE not in types
```

**Step 2: 运行测试确认失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_loop.py::TestReactLoop::test_todo_write_emits_todo_update_event -v`
Expected: FAIL — `TypeError: ReactLoop.__init__() got an unexpected keyword argument 'todo_store'`

**Step 3: 修改 `src/pure_agent_loop/loop.py`**

3a. 添加导入（文件顶部）：
```python
from .builtin_tools import TodoStore
```

3b. 修改 `ReactLoop.__init__` 签名，添加 `todo_store` 参数：
```python
    def __init__(
        self,
        llm: BaseLLMClient,
        tool_registry: ToolRegistry,
        limits: LoopLimits,
        retry: RetryConfig,
        llm_kwargs: dict[str, Any] | None = None,
        todo_store: TodoStore | None = None,
    ):
        self._llm = llm
        self._tools = tool_registry
        self._limits = limits
        self._retry_handler = RetryHandler(retry)
        self._llm_kwargs = llm_kwargs or {}
        self._todo_store = todo_store
```

3c. 在工具执行循环中（第135-153行区域），在 `yield Event.observation(...)` 之后、将工具结果追加到 `msg_history` 之前，添加 TODO_UPDATE 事件检查：

```python
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
                    msg_history.append(...)
```

**Step 4: 运行测试确认通过**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_loop.py -v`
Expected: 全部 PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/loop.py tests/test_loop.py
git commit -m "feat: ReactLoop 支持 todo_write 产出 TODO_UPDATE 事件"
```

---

## Task 5: 修改 `agent.py` — 新增 name 参数、自动注册、AgentResult.todos

**Files:**
- Modify: `src/pure_agent_loop/agent.py:1-248` (Agent 类和 AgentResult)
- Test: `tests/test_agent.py`

**Step 1: 补充测试到 `tests/test_agent.py`**

在文件末尾追加：

```python

class TestAgentName:
    """Agent name 参数测试"""

    @pytest.mark.asyncio
    async def test_default_name(self):
        """默认名称应为 '智能助理'"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        # Agent 内部应使用默认名称构建提示词
        assert agent._name == "智能助理"

    @pytest.mark.asyncio
    async def test_custom_name(self):
        """自定义名称应被保存"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm, name="研究助手")
        assert agent._name == "研究助手"


class TestAgentTodoIntegration:
    """Agent TodoWrite 集成测试"""

    @pytest.mark.asyncio
    async def test_todo_write_auto_registered(self):
        """todo_write 工具应被自动注册"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        # 工具注册表应包含 todo_write
        assert agent._tool_registry.get("todo_write") is not None

    @pytest.mark.asyncio
    async def test_agent_result_has_todos(self):
        """AgentResult 应包含 todos 属性"""
        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm)
        result = await agent.arun("打个招呼")
        assert hasattr(result, "todos")
        assert isinstance(result.todos, list)

    @pytest.mark.asyncio
    async def test_todo_write_updates_result(self):
        """通过 todo_write 工具更新的任务应出现在 AgentResult.todos"""
        mock_llm = MockLLM([
            _tool_call_response("todo_write", {
                "todos": [
                    {"content": "步骤1", "status": "completed"},
                    {"content": "步骤2", "status": "in_progress"},
                ]
            }),
            _text_response("任务已规划"),
        ])
        agent = Agent(llm=mock_llm)
        result = await agent.arun("规划任务")
        assert len(result.todos) == 2
        assert result.todos[0]["content"] == "步骤1"
        assert result.todos[1]["status"] == "in_progress"

    @pytest.mark.asyncio
    async def test_user_tools_preserved(self):
        """用户注册的工具不应被内置工具覆盖"""

        @tool
        def search(query: str) -> str:
            """搜索"""
            return "结果"

        mock_llm = MockLLM([_text_response("你好")])
        agent = Agent(llm=mock_llm, tools=[search])
        assert agent._tool_registry.get("search") is not None
        assert agent._tool_registry.get("todo_write") is not None
```

**Step 2: 运行测试确认失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_agent.py::TestAgentName -v`
Expected: FAIL — `TypeError: Agent.__init__() got an unexpected keyword argument 'name'` 或 `AttributeError: 'Agent' object has no attribute '_name'`

**Step 3: 修改 `src/pure_agent_loop/agent.py`**

3a. 添加导入（文件顶部，现有导入之后）：
```python
from .builtin_tools import TodoStore, create_todo_tool
from .prompts import build_system_prompt
```

3b. 修改 `AgentResult` — 新增 `todos` 字段：
```python
@dataclass
class AgentResult:
    """Agent 执行结果

    Attributes:
        content: 最终回答文本
        steps: 总执行步数
        total_tokens: 累计 token 用量
        events: 完整事件历史
        stop_reason: 终止原因 ("completed" | "token_limit" | "error")
        messages: 完整消息历史（可用于续接对话）
        todos: 最终任务列表
    """

    content: str
    steps: int
    total_tokens: TokenUsage
    events: list[Event]
    stop_reason: str
    messages: list[dict[str, Any]]
    todos: list[dict[str, str]] = field(default_factory=list)
```

3c. 修改 `Agent.__init__` — 新增 `name` 参数，自动创建 TodoStore 和注册 todo_write：
```python
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        llm: BaseLLMClient | None = None,
        tools: list[Tool | dict[str, Any]] | None = None,
        system_prompt: str = "",
        name: str = "智能助理",
        limits: LoopLimits | None = None,
        retry: RetryConfig | None = None,
        temperature: float = 0.7,
        **llm_kwargs: Any,
    ):
        # 构建 LLM 客户端
        if llm is not None:
            self._llm = llm
        else:
            self._llm = OpenAIClient(
                model=model,
                api_key=api_key,
                base_url=base_url,
            )

        # 创建 TodoStore 和内置工具
        self._todo_store = TodoStore()
        self._name = name

        # 注册工具（内置 + 用户）
        self._tool_registry = ToolRegistry()
        self._tool_registry.register(create_todo_tool(self._todo_store))
        if tools:
            self._tool_registry.register_many(tools)

        # 构建完整系统提示词
        self._system_prompt = build_system_prompt(
            name=name,
            user_prompt=system_prompt,
        )
        self._limits = limits or LoopLimits()
        self._retry = retry or RetryConfig()
        self._llm_kwargs: dict[str, Any] = {"temperature": temperature, **llm_kwargs}
```

3d. 修改 `_create_loop` — 传递 `todo_store`：
```python
    def _create_loop(self) -> ReactLoop:
        """创建循环引擎实例"""
        return ReactLoop(
            llm=self._llm,
            tool_registry=self._tool_registry,
            limits=self._limits,
            retry=self._retry,
            llm_kwargs=self._llm_kwargs,
            todo_store=self._todo_store,
        )
```

3e. 修改 `_build_result` — 提取 todos：
```python
    def _build_result(self, events: list[Event]) -> AgentResult:
        """从事件列表构建 AgentResult"""
        # 查找结束事件
        end_event = next(
            (e for e in events if e.type == EventType.LOOP_END),
            None,
        )

        content = ""
        stop_reason = "unknown"
        messages_history: list[dict[str, Any]] = []
        max_step = 0

        if end_event:
            content = end_event.data.get("content", "")
            stop_reason = end_event.data.get("stop_reason", "unknown")
            messages_history = end_event.data.get("messages", [])
            max_step = end_event.step

        # 累计 token（从事件推断，目前简化处理）
        total_tokens = TokenUsage.zero()

        # 从事件推断总步数
        steps = max_step

        # 提取最终 todo 状态
        todos = [t.to_dict() for t in self._todo_store.todos]

        return AgentResult(
            content=content,
            steps=steps,
            total_tokens=total_tokens,
            events=events,
            stop_reason=stop_reason,
            messages=messages_history,
            todos=todos,
        )
```

**Step 4: 运行测试确认通过**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_agent.py -v`
Expected: 全部 PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/agent.py tests/test_agent.py
git commit -m "feat: Agent 新增 name 参数、自动注册 todo_write、AgentResult.todos"
```

---

## Task 6: 修改 `renderer.py` — 新增 TODO_UPDATE 默认渲染

**Files:**
- Modify: `src/pure_agent_loop/renderer.py:80-108` (_default_render 方法)
- Test: `tests/test_renderer.py`

**Step 1: 读取并补充测试**

先读取现有 `tests/test_renderer.py`，在末尾追加：

```python

class TestRendererTodoUpdate:
    """TODO_UPDATE 事件渲染测试"""

    def test_render_todo_update(self):
        """应渲染 TODO_UPDATE 事件"""
        renderer = Renderer()
        event = Event.todo_update(
            step=1,
            todos=[
                {"content": "搜索资料", "status": "completed"},
                {"content": "分析结果", "status": "in_progress"},
                {"content": "撰写报告", "status": "pending"},
            ],
        )
        output = renderer.render(event)
        assert "搜索资料" in output
        assert "分析结果" in output
        assert "撰写报告" in output
```

**Step 2: 运行测试确认失败**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_renderer.py::TestRendererTodoUpdate -v`
Expected: FAIL — match 语句中无 TODO_UPDATE case

**Step 3: 修改 `src/pure_agent_loop/renderer.py`**

在 `_default_render` 方法的 match 语句中，`LOOP_END` case 之后、`_` case 之前，添加：

```python
            case EventType.TODO_UPDATE:
                todos = event.data.get("todos", [])
                if not todos:
                    return "📋 任务列表为空"
                icons = {"pending": "⬜", "in_progress": "🔄", "completed": "✅"}
                lines = ["📋 任务进度更新："]
                for i, t in enumerate(todos, 1):
                    icon = icons.get(t.get("status", ""), "❓")
                    lines.append(f"  {i}. {icon} {t.get('content', '')}")
                completed = sum(1 for t in todos if t.get("status") == "completed")
                lines.append(f"[{completed}/{len(todos)} 完成]")
                return "\n".join(lines)
```

**Step 4: 运行测试确认通过**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest tests/test_renderer.py -v`
Expected: 全部 PASS

**Step 5: 提交**

```bash
git add src/pure_agent_loop/renderer.py tests/test_renderer.py
git commit -m "feat: Renderer 新增 TODO_UPDATE 事件默认渲染"
```

---

## Task 7: 修改 `__init__.py` — 导出新增公共 API

**Files:**
- Modify: `src/pure_agent_loop/__init__.py`

**Step 1: 修改导出**

添加导入行：
```python
from .builtin_tools import TodoItem, TodoStore
from .prompts import build_system_prompt
```

在 `__all__` 中添加：
```python
    # 内置工具
    "TodoItem",
    "TodoStore",
    # 提示词
    "build_system_prompt",
```

**Step 2: 运行全部测试确认无破坏**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest -v`
Expected: 全部 PASS

**Step 3: 提交**

```bash
git add src/pure_agent_loop/__init__.py
git commit -m "feat: 导出 TodoItem, TodoStore, build_system_prompt"
```

---

## Task 8: 更新示例代码

**Files:**
- Modify: `examples/basic.py`
- Modify: `examples/streaming.py`

**Step 1: 更新 `examples/basic.py`**

修改 Agent 构造处使用 name 参数：
```python
def main():
    agent = Agent(
        name="研究助手",
        model=os.getenv("MODEL", "deepseek-chat"),
        api_key=os.environ["API_KEY"],
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        tools=[search, calculate],
        system_prompt="你擅长搜索信息和计算数学表达式。",
    )

    result = agent.run("Python 语言是什么时候发布的？1991 年到 2026 年一共多少年？")
    print(f"回答: {result.content}")
    print(f"步数: {result.steps}")
    print(f"终止原因: {result.stop_reason}")

    # 展示任务追踪结果
    if result.todos:
        print("\n📋 任务追踪：")
        for todo in result.todos:
            print(f"  [{todo['status']}] {todo['content']}")
```

**Step 2: 更新 `examples/streaming.py`**

在异步流式输出部分添加 TODO_UPDATE 事件的专门处理：
```python
from pure_agent_loop import Agent, tool, Renderer, EventType

async def main():
    agent = Agent(
        name="搜索助手",
        model=os.getenv("MODEL", "deepseek-chat"),
        api_key=os.environ["API_KEY"],
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        tools=[search],
        system_prompt="你擅长搜索网络信息。",
    )

    renderer = Renderer()

    # 异步流式执行，包含任务进度实时输出
    async for event in agent.arun_stream("搜索 Python 最新版本信息"):
        output = renderer.render(event)
        if output:
            print(output)
```

**Step 3: 提交**

```bash
git add examples/basic.py examples/streaming.py
git commit -m "docs: 更新示例代码展示 name 参数和 TodoWrite 功能"
```

---

## Task 9: 最终验证 — 全量测试 + 覆盖率

**Step 1: 运行全部测试**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest -v`
Expected: 全部 PASS

**Step 2: 运行覆盖率检查**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && pytest --cov=pure_agent_loop --cov-report=term-missing`
Expected: 新模块 (prompts.py, builtin_tools.py) 覆盖率 >= 80%

**Step 3: 检查导入**

Run: `cd /Users/xziying/project/github/pure-agent-loop && source venv/bin/activate && python -c "from pure_agent_loop import Agent, TodoItem, TodoStore, build_system_prompt; print('导入成功')"``
Expected: 输出 "导入成功"
