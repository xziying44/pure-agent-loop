"""多轮对话示例: 使用 Conversation 实现会话续接

使用前请先安装依赖并配置环境变量:
    pip install python-dotenv
    cp .env.example .env
    # 编辑 .env 填入实际的 API 密钥
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from pure_agent_loop import Agent, Renderer

# 加载 examples/.env 配置
load_dotenv(Path(__file__).parent / ".env")


async def main():
    agent = Agent(
        name="对话助手",
        model=os.getenv("MODEL", "deepseek-chat"),
        api_key=os.environ["API_KEY"],
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        system_prompt="你是一个友好的对话助手。",
    )
    renderer = Renderer()

    # 创建会话 — 多轮对话自动续接
    conv = agent.conversation()

    print("=" * 50)
    print("第一轮对话")
    print("=" * 50)

    async for event in conv.send_stream("Python 是什么语言？"):
        output = renderer.render(event)
        if output:
            print(output)

    print(f"\n当前消息历史长度: {len(conv.messages)}")

    print("\n" + "=" * 50)
    print("第二轮对话（自动续接上下文）")
    print("=" * 50)

    async for event in conv.send_stream("它有哪些主要的应用领域？"):
        output = renderer.render(event)
        if output:
            print(output)

    print(f"\n当前消息历史长度: {len(conv.messages)}")

    # 创建全新对话（不带历史）
    print("\n" + "=" * 50)
    print("全新对话（独立 Conversation）")
    print("=" * 50)

    conv2 = agent.conversation()
    async for event in conv2.send_stream("1 + 1 等于多少？"):
        output = renderer.render(event)
        if output:
            print(output)


if __name__ == "__main__":
    asyncio.run(main())
