"""自定义 LLM 客户端示例: 接入非 OpenAI 兼容的模型"""

from typing import Any
from pure_agent_loop import Agent, BaseLLMClient, LLMResponse, ToolCall, TokenUsage


class MyCustomLLM(BaseLLMClient):
    """自定义 LLM 客户端示例

    演示如何接入任意 LLM API。
    只需继承 BaseLLMClient 并实现 chat 方法。
    """

    def __init__(self, model: str):
        self.model = model

    async def chat(
        self,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        **kwargs: Any,
    ) -> LLMResponse:
        # 这里替换为你的 LLM API 调用逻辑
        # 示例: 返回一个简单的响应
        last_message = messages[-1]["content"]

        return LLMResponse(
            content=f"[{self.model}] 收到消息: {last_message}",
            tool_calls=[],
            usage=TokenUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
            raw={},
        )


def main():
    # 使用自定义客户端
    custom_llm = MyCustomLLM(model="my-custom-model")
    agent = Agent(llm=custom_llm)

    result = agent.run("你好")
    print(f"回答: {result.content}")


if __name__ == "__main__":
    main()
