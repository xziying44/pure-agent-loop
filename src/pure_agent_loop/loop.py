"""ReAct 循环引擎

实现 Thought → Action → Observation 的循环流程。
"""

import json
import logging
import time
from typing import Any, AsyncIterator

from .events import Event, EventType
from .limits import LoopLimits, LimitChecker
from .llm.base import BaseLLMClient
from .llm.types import LLMResponse
from .retry import RetryConfig, RetryHandler
from .tool import ToolRegistry

logger = logging.getLogger(__name__)


class ReactLoop:
    """ReAct 循环引擎

    执行 Thought → Action → Observation 循环，直到满足终止条件。

    Args:
        llm: LLM 客户端实例
        tool_registry: 工具注册表
        limits: 终止条件配置
        retry: 重试配置
    """

    def __init__(
        self,
        llm: BaseLLMClient,
        tool_registry: ToolRegistry,
        limits: LoopLimits,
        retry: RetryConfig,
        llm_kwargs: dict[str, Any] | None = None,
    ):
        self._llm = llm
        self._tools = tool_registry
        self._limits = limits
        self._retry_handler = RetryHandler(retry)
        self._llm_kwargs = llm_kwargs or {}

    async def run(
        self,
        task: str,
        system_prompt: str = "You are a helpful assistant.",
        messages: list[dict[str, Any]] | None = None,
    ) -> AsyncIterator[Event]:
        """执行 ReAct 循环

        Args:
            task: 用户任务描述
            system_prompt: 系统提示
            messages: 初始消息历史（用于多轮对话）

        Yields:
            Event: 执行过程中的结构化事件
        """
        # 初始化消息历史
        if messages:
            msg_history = list(messages)
            msg_history.append({"role": "user", "content": task})
        else:
            msg_history = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

        # 初始化限制检查器
        checker = LimitChecker(self._limits)

        # 获取工具 schema
        tool_schemas = self._tools.to_openai_schemas() or None

        # 产出循环启动事件
        yield Event.loop_start(task=task)

        step = 0
        final_content = ""

        while True:
            step += 1
            checker.current_step = step

            # ---- 调用 LLM ----
            try:
                response = await self._retry_handler.execute(
                    self._llm.chat,
                    messages=msg_history,
                    tools=tool_schemas,
                    **self._llm_kwargs,
                )
            except Exception as e:
                logger.error("LLM 调用失败 (重试耗尽): %s", e)
                yield Event.error(step=step, error=str(e), fatal=True)
                yield Event.loop_end(
                    step=step,
                    stop_reason="error",
                    content=f"LLM 调用失败: {e}",
                    messages=msg_history,
                )
                return

            # 累加 token
            checker.add_tokens(response.usage.total_tokens)

            # ---- 处理响应 ----
            if response.has_tool_calls:
                # Thought（如果有文本内容）
                if response.content:
                    yield Event.thought(step=step, content=response.content)

                # 构建 assistant 消息
                assistant_msg: dict[str, Any] = {"role": "assistant"}
                if response.content:
                    assistant_msg["content"] = response.content
                assistant_msg["tool_calls"] = [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.name,
                            "arguments": json.dumps(tc.arguments),
                        },
                    }
                    for tc in response.tool_calls
                ]
                msg_history.append(assistant_msg)

                # ---- 执行工具 ----
                for tc in response.tool_calls:
                    yield Event.action(step=step, tool=tc.name, args=tc.arguments)

                    start_time = time.time()
                    result = await self._tools.execute(tc.name, tc.arguments)
                    duration = time.time() - start_time

                    yield Event.observation(
                        step=step, tool=tc.name, result=result, duration=duration
                    )

                    # 将工具结果追加到消息历史
                    msg_history.append(
                        {
                            "role": "tool",
                            "tool_call_id": tc.id,
                            "content": result,
                        }
                    )

            else:
                # 无工具调用 = 最终回答
                final_content = response.content or ""
                if final_content:
                    yield Event.thought(step=step, content=final_content)

                msg_history.append(
                    {"role": "assistant", "content": final_content}
                )

                yield Event.loop_end(
                    step=step,
                    stop_reason="completed",
                    content=final_content,
                    messages=msg_history,
                )
                return

            # ---- 检查限制 ----
            limit_result = checker.check()

            if limit_result.action == "stop":
                yield Event.loop_end(
                    step=step,
                    stop_reason=limit_result.reason,
                    content=final_content,
                    messages=msg_history,
                )
                return

            if limit_result.action == "warn":
                yield Event.soft_limit(
                    step=step,
                    reason=limit_result.reason,
                    prompt=limit_result.prompt,
                )
                # 注入系统提示
                msg_history.append(
                    {"role": "system", "content": limit_result.prompt}
                )
