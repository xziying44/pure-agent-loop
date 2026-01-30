"""基础示例: 最简 Agent 使用

使用前请先安装依赖并配置环境变量:
    pip install python-dotenv
    cp .env.example .env
    # 编辑 .env 填入实际的 API 密钥
"""

import os
from pathlib import Path

from dotenv import load_dotenv

from pure_agent_loop import Agent, tool

# 加载 examples/.env 配置
load_dotenv(Path(__file__).parent / ".env")


@tool
def search(query: str) -> str:
    """搜索网页内容

    Args:
        query: 搜索关键词
    """
    # 这里替换为实际的搜索实现
    return f"搜索 '{query}' 的结果: Python 是一种通用编程语言..."


@tool
def calculate(expression: str) -> str:
    """计算数学表达式

    Args:
        expression: 数学表达式
    """
    try:
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


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


if __name__ == "__main__":
    main()
