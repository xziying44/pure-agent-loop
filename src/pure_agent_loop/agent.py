"""Agent 入口

用户使用 pure-agent-loop 的唯一入口点。
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Iterator, Literal

from .events import Event, EventType
from .limits import LoopLimits
from .llm.base import BaseLLMClient
from .llm.openai_client import OpenAIClient
from .llm.types import TokenUsage
from .loop import ReactLoop
from .retry import RetryConfig
from .tool import Tool, ToolRegistry
from .builtin_tools import TodoStore, create_todo_tool
from .prompts import build_system_prompt

logger = logging.getLogger(__name__)

# 思考深度类型
ThinkingLevel = Literal["off", "low", "medium", "high"]


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


class Agent:
    """Agent 入口类

    使用 pure-agent-loop 的唯一入口。

    使用方式 1 - 通过 model 参数（内置 OpenAI 兼容客户端）:
        agent = Agent(model="deepseek-chat", api_key="sk-xxx", base_url="...")

    使用方式 2 - 通过自定义 LLM 客户端:
        agent = Agent(llm=MyCustomClient())

    Args:
        model: 模型名称（使用内置客户端时）
        api_key: API 密钥（默认读取环境变量）
        base_url: API 基础地址
        llm: 自定义 LLM 客户端实例（与 model 二选一）
        tools: 工具列表（@tool 装饰器或字典格式）
        system_prompt: 系统提示
        limits: 终止条件配置
        retry: 重试配置
        temperature: 温度参数
        thinking_level: 思考深度（off/low/medium/high），默认 off
        emit_reasoning_events: 是否推送 REASONING 事件，默认 False
        **llm_kwargs: 透传给 LLM 调用的额外参数
    """

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
        thinking_level: ThinkingLevel = "off",
        emit_reasoning_events: bool = False,
        **llm_kwargs: Any,
    ):
        # 保存思考模式配置
        self._thinking_level = thinking_level
        self._emit_reasoning_events = emit_reasoning_events

        # 构建 LLM 客户端
        if llm is not None:
            self._llm = llm
        else:
            self._llm = OpenAIClient(
                model=model,
                api_key=api_key,
                base_url=base_url,
                thinking_level=thinking_level,
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

    def _create_loop(self) -> ReactLoop:
        """创建循环引擎实例"""
        return ReactLoop(
            llm=self._llm,
            tool_registry=self._tool_registry,
            limits=self._limits,
            retry=self._retry,
            llm_kwargs=self._llm_kwargs,
            todo_store=self._todo_store,
            emit_reasoning_events=self._emit_reasoning_events,
        )

    async def arun_stream(
        self,
        task: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        """异步流式执行

        Args:
            task: 任务描述
            messages: 初始消息历史（多轮对话续接）

        Yields:
            Event: 执行过程中的结构化事件
        """
        loop = self._create_loop()
        async for event in loop.run(
            task=task,
            system_prompt=self._system_prompt,
            messages=messages,
        ):
            yield event

    async def arun(
        self,
        task: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> AgentResult:
        """异步执行（阻塞等待最终结果）

        Args:
            task: 任务描述
            messages: 初始消息历史

        Returns:
            AgentResult: 执行结果
        """
        events: list[Event] = []
        async for event in self.arun_stream(task, messages=messages):
            events.append(event)

        return self._build_result(events)

    def run_stream(
        self,
        task: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> Iterator[Event]:
        """同步流式执行

        Args:
            task: 任务描述
            messages: 初始消息历史

        Yields:
            Event: 执行过程中的结构化事件
        """
        # 检查是否已有运行的事件循环
        try:
            asyncio.get_running_loop()
            # 已有事件循环在运行，需要在新线程中执行
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                new_loop = asyncio.new_event_loop()

                async def _collect():
                    events = []
                    async for event in self.arun_stream(task, messages=messages):
                        events.append(event)
                    return events

                future = pool.submit(new_loop.run_until_complete, _collect())
                events = future.result()
                new_loop.close()
        except RuntimeError:
            # 无运行的事件循环，可以直接运行

            async def _collect():
                events = []
                async for event in self.arun_stream(task, messages=messages):
                    events.append(event)
                return events

            events = asyncio.run(_collect())

        yield from events

    def run(
        self,
        task: str,
        messages: list[dict[str, Any]] | None = None,
    ) -> AgentResult:
        """同步执行（阻塞等待最终结果）

        Args:
            task: 任务描述
            messages: 初始消息历史

        Returns:
            AgentResult: 执行结果
        """
        events = list(self.run_stream(task, messages=messages))
        return self._build_result(events)

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
