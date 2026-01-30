"""使用 Skill 系统的 Exa 搜索示例

这个示例展示了如何使用 Skill 系统来指导智能体的工具使用策略。
与原版 exa_search.py 相比，系统提示词更简洁，工具使用策略通过 Skill 动态加载。
"""

import asyncio
import os
import re
from pathlib import Path

import html2text
import requests
from dotenv import load_dotenv

from pure_agent_loop import Agent, ThinkingLevel, tool

# 尝试导入 rich_renderer（如果存在）
try:
    from rich_renderer import RichRenderer

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


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

        results = data.get("results", [])
        if not results:
            return f"未找到关于 '{query}' 的搜索结果"

        formatted = []
        for i, result in enumerate(results, 1):
            title = result.get("title", "无标题")
            url = result.get("url", "")
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
def fetch_webpage(url: str, max_length: int = 8000) -> str:
    """访问指定 URL 并将网页内容转换为 Markdown 格式返回

    Args:
        url: 要访问的网页 URL（必须是完整的 http/https 地址）
        max_length: 返回内容的最大字符数，默认 8000
    """
    if not url.startswith(("http://", "https://")):
        return "错误: URL 必须以 http:// 或 https:// 开头"

    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/120.0.0.0 Safari/537.36"
            ),
        }

        response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
        response.raise_for_status()
        response.encoding = response.apparent_encoding or "utf-8"
        html_content = response.text

        converter = html2text.HTML2Text()
        converter.ignore_links = False
        converter.ignore_images = True
        converter.body_width = 0

        markdown_content = converter.handle(html_content)
        markdown_content = re.sub(r"\n{3,}", "\n\n", markdown_content).strip()

        if len(markdown_content) > max_length:
            markdown_content = markdown_content[:max_length] + "\n\n... [内容已截断]"

        return f"# 网页内容 ({url})\n\n{markdown_content}"

    except Exception as e:
        return f"错误: {e}"


async def main():
    if not os.getenv("API_KEY"):
        print("错误: 请在 .env 文件中配置 API_KEY")
        return
    if not EXA_API_KEY:
        print("错误: 请在 .env 文件中配置 EXA_API_KEY")
        return

    thinking_level: ThinkingLevel = os.getenv("THINKING_LEVEL", "off")  # type: ignore

    agent = Agent(
        name="搜索助手",
        model=os.getenv("MODEL", "deepseek-chat"),
        api_key=os.environ["API_KEY"],
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        tools=[exa_search, fetch_webpage],
        skills_dir=str(Path(__file__).parent / "skills"),
        system_prompt=(
            "你是一个专业的助手。\n\n"
            "工作原则：\n"
            "1. 如果任务涉及网络搜索或信息检索，请先使用 skill 工具加载 'web-search' 技能获取详细指导\n"
            "2. 根据技能指导合理使用工具\n"
            "3. 注明信息来源"
        ),
        thinking_level=thinking_level,
        emit_reasoning_events=thinking_level != "off",
    )

    query = "我想了解一下灵迹岛这款桌游的规则"
    print(f"\n🔍 查询: {query}\n")
    print("=" * 60)

    if HAS_RICH:
        renderer = RichRenderer(
            max_thought_lines=3,
            max_result_chars=150,
        )
        async for event in agent.arun_stream(query):
            renderer.render(event)
    else:
        async for event in agent.arun_stream(query):
            print(f"[{event.type.value}] {event.data}")

    print("\n" + "=" * 60)
    print("✅ 执行完成")


if __name__ == "__main__":
    asyncio.run(main())

