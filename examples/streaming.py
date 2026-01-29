"""流式输出示例: 实时查看 Agent 执行过程"""

import asyncio
from pure_agent_loop import Agent, tool, Renderer


@tool
def search(query: str) -> str:
    """搜索网页

    Args:
        query: 搜索关键词
    """
    return f"搜索结果: {query} 相关信息..."


async def main():
    agent = Agent(
        model="deepseek-chat",
        api_key="your-api-key",
        base_url="https://api.deepseek.com/v1",
        tools=[search],
    )

    renderer = Renderer()

    # 异步流式执行
    async for event in agent.arun_stream("搜索 Python 最新版本信息"):
        output = renderer.render(event)
        if output:
            print(output)

    print("\n--- 同步方式 ---\n")

    # 同步流式执行
    for event in agent.run_stream("搜索 Python 最新版本信息"):
        print(event.to_dict())


if __name__ == "__main__":
    asyncio.run(main())
