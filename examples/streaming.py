"""流式输出示例: 实时查看 Agent 执行过程

使用前请先安装依赖并配置环境变量:
    pip install python-dotenv
    cp .env.example .env
    # 编辑 .env 填入实际的 API 密钥
"""

import asyncio
import os
from pathlib import Path

from dotenv import load_dotenv

from pure_agent_loop import Agent, tool, Renderer, EventType

# 加载 examples/.env 配置
load_dotenv(Path(__file__).parent / ".env")


@tool
def search(query: str) -> str:
    """搜索网页

    Args:
        query: 搜索关键词
    """
    return f"搜索结果: {query} 相关信息..."


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

    print("\n--- 同步方式 ---\n")


if __name__ == "__main__":
    asyncio.run(main())
