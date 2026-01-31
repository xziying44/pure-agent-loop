"""使用类工具注册的 Exa 搜索示例

演示如何将多个相关工具组织在一个类中，通过 @tool 装饰器标记方法，
由框架自动扫描并注册。与独立函数方式相比，类方式提供：
- 工具逻辑分组（同一类的工具共享 tool_prefix 前缀）
- 共享状态（如 API 配置、HTTP 客户端）
- 更清晰的代码组织
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


class ExaWebTools:
    """Exa 搜索 & 网页抓取工具集

    将搜索和网页抓取两个工具组织在同一个类中，
    共享 API 配置和 HTTP 请求头等状态。
    """

    tool_prefix = "exa"

    def __init__(
        self,
        api_key: str = "",
        api_url: str = "https://api.exa.ai/search",
        request_timeout: int = 30,
    ):
        self.api_key = api_key
        self.api_url = api_url
        self.timeout = request_timeout
        self.user_agent = (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/120.0.0.0 Safari/537.36"
        )

    @tool
    def search(self, query: str, num_results: int = 5) -> str:
        """使用 Exa AI 搜索网页内容

        Args:
            query: 搜索查询词
            num_results: 返回结果数量，默认5条
        """
        if not self.api_key:
            return "错误: 未配置 EXA_API_KEY 环境变量"

        try:
            response = requests.post(
                self.api_url,
                headers={
                    "Content-Type": "application/json",
                    "x-api-key": self.api_key,
                },
                json={
                    "query": query,
                    "type": "auto",
                    "numResults": num_results,
                    "contents": {"text": True},
                },
                timeout=self.timeout,
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
    def fetch_webpage(self, url: str, max_length: int = 8000) -> str:
        """访问指定 URL 并将网页内容转换为 Markdown 格式返回

        Args:
            url: 要访问的网页 URL（必须是完整的 http/https 地址）
            max_length: 返回内容的最大字符数，默认 8000
        """
        if not url.startswith(("http://", "https://")):
            return "错误: URL 必须以 http:// 或 https:// 开头"

        try:
            response = requests.get(
                url,
                headers={"User-Agent": self.user_agent},
                timeout=self.timeout,
                allow_redirects=True,
            )
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

    exa_api_key = os.getenv("EXA_API_KEY", "")
    if not exa_api_key:
        print("错误: 请在 .env 文件中配置 EXA_API_KEY")
        return

    thinking_level: ThinkingLevel = os.getenv("THINKING_LEVEL", "off")  # type: ignore

    # 创建工具类实例，注入 API 配置
    exa_tools = ExaWebTools(api_key=exa_api_key)

    agent = Agent(
        name="搜索助手",
        model=os.getenv("MODEL", "deepseek-chat"),
        api_key=os.environ["API_KEY"],
        base_url=os.getenv("BASE_URL", "https://api.deepseek.com/v1"),
        tools=[exa_tools],  # 直接传入类实例，框架自动扫描注册
        skills_dir=str(Path(__file__).parent / "skills"),
        system_prompt="你是一个专业的信息检索助手，擅长搜索和整理网络信息。回答时请注明信息来源。",
        thinking_level=thinking_level,
        emit_reasoning_events=thinking_level != "off",
    )

    # 注册后的工具名称：exa_search, exa_fetch_webpage

    query = "灵迹岛中放置污染的时候，是所有人的灵迹都被摧毁，还是只摧毁一个？"
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
