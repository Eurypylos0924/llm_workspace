# external_adapter/tavily_search.py(전용 검색 모듈)

import os
from typing import Optional
from tavily import TavilyClient


class TavilySearchEngine:
    """
    Tavily API를 사용해 실시간 웹 검색을 수행하는 외부 검색 엔진 클래스입니다.
    """

    def __init__(self, api_key: Optional[str] = None):
        # 전달받은 api_key가 없으면 환경변수 TAVILY_API_KEY 참조
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        self.client = TavilyClient(api_key=self.api_key) if self.api_key else None

    def search(self, query: str, max_results: int = 3) -> str:
        """
        검색 키워드를 입력받아 Tavily API를 호출하고 결과를 문자열로 정형화하여 반환합니다.

        :param query: 검색어
        :param max_results: 반환받을 검색 결과 최대 수
        :return: 검색 결과 요약 텍스트
        """
        if not self.client:
            return "에러: TAVILY_API_KEY가 설정되어 있지 않습니다."

        try:
            response = self.client.search(query=query, max_results=max_results)
            results = response.get("results", [])

            if not results:
                return "검색 결과가 없습니다."

            formatted_results = []
            for item in results:
                title = item.get("title", "")
                content = item.get("content", "")
                url = item.get("url", "")
                formatted_results.append(f"제목: {title}\n내용: {content}\n출처: {url}\n")

            return "\n---\n".join(formatted_results)
        except Exception as e:
            return f"Tavily 검색 실행 중 에러 발생: {str(e)}"