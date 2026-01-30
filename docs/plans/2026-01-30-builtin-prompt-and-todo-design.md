# 内置系统提示词 & TodoWrite 工具设计方案

**日期**: 2026-01-30
**状态**: 已批准

## 概述

为 `pure-agent-loop` 框架新增内置系统提示词模板和 TodoWrite 任务管理工具，使智能体具备结构化的 ReAct 行为规范和自动任务追踪能力。

## 设计决策

| 决策项 | 选择 | 说明 |
|--------|------|------|
| 提示词模式 | 模板包裹模式 | 内置模板定义框架行为，用户 system_prompt 注入模板中的固定区域 |
| 响应格式 | 自然思考模式 | LLM content 作为思考，tool_calls 作为行动，与 Function Calling 协议契合 |
| TodoWrite 集成 | 内置自动注册 | 框架自动注册 todo_write 工具，用户无需手动添加 |
| Todo 存储 | 内存状态 | Todo 保存在 Agent 实例内存中，通过 AgentResult.todos 返回 |
| 名称注入 | 新增 name 参数 | Agent(name="xxx")，默认 "智能助理" |
| 事件系统 | 新增 TODO_UPDATE | 流式输出中包含 todo 变更事件 |

## API 变更

### Agent 构造函数

```python
class Agent:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        llm: BaseLLMClient | None = None,
        tools: list[Tool | dict[str, Any]] | None = None,
        system_prompt: str = "",          # 用户自定义任务指令（注入模板）
        name: str = "智能助理",           # 新增：智能体名称
        limits: LoopLimits | None = None,
        retry: RetryConfig | None = None,
        temperature: float = 0.7,
        **llm_kwargs: Any,
    ):
```

### AgentResult 新增属性

```python
@dataclass
class AgentResult:
    # ... 原有字段 ...
    todos: list[dict[str, str]]  # 新增：最终 todo 列表
```

## 新增模块

### prompts.py — 内置系统提示词模板

- `build_system_prompt(name, user_prompt)` 函数
- 模板包含：角色描述、ReAct 行为规范、TodoWrite 使用要求、用户自定义指令区域
- name 默认值 "智能助理"

### builtin_tools.py — TodoWrite 内置工具

- `TodoItem` 数据类：content + status
- `TodoStore` 类：内存状态管理，write() 方法完全替换列表
- `create_todo_tool(store)` 工厂函数：创建绑定到特定 store 的 Tool 实例
- 工具返回值包含格式化的完整 todo 列表（注入 LLM 上下文）

## 修改模块

### events.py
- EventType 新增 `TODO_UPDATE = "todo_update"`
- Event 新增 `todo_update()` 工厂方法

### agent.py
- 新增 `name` 参数
- 自动创建 `TodoStore` 和 `todo_write` 工具
- 调用 `build_system_prompt()` 生成完整提示词
- `AgentResult` 新增 `todos` 字段
- `_build_result()` 提取最终 todo 状态

### loop.py
- `ReactLoop.__init__` 接受 `todo_store` 参数
- 工具执行后检查是否为 `todo_write`，额外 yield TODO_UPDATE 事件

### __init__.py
- 导出 `TodoItem`, `TodoStore`

## 文件变更清单

| 操作 | 文件 |
|------|------|
| 新增 | `src/pure_agent_loop/prompts.py` |
| 新增 | `src/pure_agent_loop/builtin_tools.py` |
| 修改 | `src/pure_agent_loop/events.py` |
| 修改 | `src/pure_agent_loop/agent.py` |
| 修改 | `src/pure_agent_loop/loop.py` |
| 修改 | `src/pure_agent_loop/__init__.py` |
| 新增 | `tests/test_prompts.py` |
| 新增 | `tests/test_builtin_tools.py` |
| 修改 | `tests/test_agent.py` |
| 修改 | `tests/test_loop.py` |
| 修改 | `examples/basic.py` |
| 修改 | `examples/streaming.py` |
