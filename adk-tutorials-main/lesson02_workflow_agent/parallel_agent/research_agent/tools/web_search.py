import os

from dotenv import load_dotenv
from tavily import TavilyClient

load_dotenv(override=True)

# OpenAI 내장 web_search 도구는 NVIDIA 모델에서 쓸 수 없으므로 Tavily 검색 API로 대체
client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def tavily_web_search(query: str) -> str:
    """
    Tavily Web Search를 이용하여 인터넷에서 최신 정보를 검색한다.

    Args:
        query: 검색할 검색어

    Returns:
        검색 결과 텍스트
    """

    try:
        response = client.search(query=query, max_results=3)

        return "\n\n".join(
            f"- {r['title']}\n  {r['content']}\n  ({r['url']})"
            for r in response.get("results", [])
        ) or "검색 결과가 없습니다."

    except Exception as e:
        return f"웹 검색 중 오류가 발생했습니다: {e}"
