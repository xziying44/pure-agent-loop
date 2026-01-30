"""Exa 搜索示例: 使用 Exa AI 进行网页搜索（流式事件输出）

使用前请先安装依赖并配置环境变量:
    pip install python-dotenv requests rich
    cp .env.example .env
    # 编辑 .env 填入 API_KEY 和 EXA_API_KEY
"""

import asyncio
import os
from pathlib import Path

import requests
from dotenv import load_dotenv

from pure_agent_loop import Agent, tool
from rich_renderer import RichRenderer

# 加载 examples/.env 配置
load_dotenv(Path(__file__).parent / ".env")

# Exa API 配置
EXA_API_URL = "https://api.exa.ai/search"
EXA_API_KEY = os.getenv("EXA_API_KEY", "")


@tool
def exa_search(query: str, num_results: int = 5) -> str:
    """使用 Exa AI 搜索网页内容

    Args:
        query: 搜索查询词
        num_results: 返回结果数量，默认5条
    """
    if not EXA_API_KEY:
        return "错误: 未配置 EXA_API_KEY 环境变量"

    try:
        response = requests.post(
            EXA_API_URL,
            headers={
                "Content-Type": "application/json",
                "x-api-key": EXA_API_KEY,
            },
            json={
                "query": query,
                "type": "auto",
                "numResults": num_results,
                "contents": {"text": True},
            },
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        # 格式化搜索结果
        results = data.get("results", [])
        if not results:
            return f"未找到关于 '{query}' 的搜索结果"

        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "无标题")
            url = result.get("url", "")
            # 截取文本内容前500字符
            text = result.get("text", "")[:500]
            if len(result.get("text", "")) > 500:
                text += "..."

            formatted.append(f"[{i}] {title}\n    URL: {url}\n    内容: {text}\n")

        return "\n".join(formatted)

    except requests.exceptions.Timeout:
        return "错误: 搜索请求超时"
    except requests.exceptions.RequestException as e:
        return f"错误: 搜索请求失败 - {e}"
    except Exception as e:
        return f"错误: {e}"


@tool
def calculate(expression: str) -> str:
    """计算数学表达式

    Args:
        expression: 数学表达式
    """
    try:
        # 安全地计算数学表达式
        allowed = set("0123456789+-*/().% ")
        if not all(c in allowed for c in expression):
            return "错误: 表达式包含不允许的字符"
        return str(eval(expression))
    except Exception as e:
        return f"计算错误: {e}"


async def main():
    # 检查必要的配置
    if not os.getenv("API_KEY"):
        print("错误: 请在 .env 文件中配置 API_KEY")
        return
    if not EXA_API_KEY:
        print("错误: 请在 .env 文件中配置 EXA_API_KEY")
        return

    agent = Agent(
        name="搜索助手",
        model=os.getenv("MODEL", "deepseek-chat"),
        api_key=os.environ["API_KEY"],
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        tools=[exa_search, calculate],
        system_prompt="你是一个专业的搜索助手。当用户询问信息时，使用 exa_search 工具搜索最新的网页内容，并根据搜索结果回答问题。回答时请注明信息来源。",
    )

    # 测试查询
    query = "我想了解智能体的skill是啥原理，这个东西最近为啥这么火"
    print(f"\n🔍 查询: {query}\n")
    print("=" * 60)

    # 使用 RichRenderer 美化输出
    renderer = RichRenderer(
        max_thought_lines=3,      # 思考内容最多显示3行
        max_result_chars=150,     # 工具结果最多显示150字符
        show_todo_table=True,     # 用表格显示 Todo 列表
    )

    # 流式执行，实时输出事件
    async for event in agent.arun_stream(query):
        renderer.render(event)

    print("\n" + "=" * 60)
    print("✅ 执行完成")


if __name__ == "__main__":
    asyncio.run(main())
