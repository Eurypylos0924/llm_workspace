# mcp_server/server.py(어댑터 연결 및 Tool 등록)


import sys
import os
import platform
import datetime
from dotenv import load_dotenv
from mcp.server.mcpserver import MCPServer
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI


# 외부 샌드박스 엔진 임포트
from external_adapter.python_sandbox import SafePythonSandbox
from external_adapter.weather import WeatherAdapter
from external_adapter.tavily_search import TavilySearchEngine
from external_adapter.google_calendar import GoogleCalendarAdapter
from external_adapter.vector_db import VectorDBAdapter

# .env 환경변수 로드
load_dotenv(override=True, dotenv_path='.env')

# MCPServer 인스턴스 생성
mcp = MCPServer("calculator-search-and-resources, google-calendar-service")


# 외부 엔진 인스턴스 생성
sandbox = SafePythonSandbox()
weather_adapter = WeatherAdapter()
search_engine = TavilySearchEngine()
calendar_adapter = GoogleCalendarAdapter()
vector_db_adapter = VectorDBAdapter()

# ==========================================
# 🛠️ MCP Tools (도구)
# ==========================================

@mcp.tool()
def safe_calculate(expression: str) -> str:
    """수식 문자열을 받아 안전한 AST 샌드박스 엔진으로 계산 결과를 반환합니다.

    Args:
        expression (str): 평가할 수식 문자열 (예: '(12 + 8) * 3')

    Returns:
        str: 연산 결과값 또는 에러 메시지
    """
    try:
        result = sandbox.calculate(expression)
        return str(result)
    except Exception as e:
        return f"계산 오류: {str(e)}"


@mcp.tool()
def get_weather_forecast(latitude: float = 37.5665, longitude: float = 126.9780) -> str:
    """Open-Meteo API 기반으로 지정한 위치(위도, 경도)의 실시간 및 오늘 예보 날씨 정보를 조회합니다.
    (기본값: 서울 - 위도 37.5665, 경도 126.9780)

    Args:
        latitude (float): 위도 (예: 서울 37.5665, 부산 35.1796)
        longitude (float): 경도 (예: 서울 126.9780, 부산 129.0756)

    Returns:
        str: 현재 기온, 습도, 풍속 및 최고/최저 기온 정보
    """
    try:
        data = weather_adapter.get_forecast(latitude=latitude, longitude=longitude)
        current = data.get("current", {})
        daily = data.get("daily", {})

        temp = current.get("temperature_2m")
        humidity = current.get("relative_humidity_2m")
        wind = current.get("wind_speed_10m")

        max_temp = daily.get("temperature_2m_max", [None])[0]
        min_temp = daily.get("temperature_2m_min", [None])[0]
        rain_prob = daily.get("precipitation_probability_max", [None])[0]

        return (
            f"🌦️ [날씨 정보 (위도: {latitude}, 경도: {longitude})]\n"
            f"- 현재 기온: {temp}°C\n"
            f"- 상대 습도: {humidity}%\n"
            f"- 풍속: {wind} km/h\n"
            f"- 오늘 최고/최저 기온: {max_temp}°C / {min_temp}°C\n"
            f"- 강수 확률: {rain_prob}%"
        )
    except Exception as e:
        return f"날씨 정보 조회 중 오류가 발생했습니다: {str(e)}"

@mcp.tool()
def web_search(query: str) -> str:
    """Tavily API 전용 엔진을 사용하여 실시간 웹 검색을 수행하고 결과를 반환합니다.

    Args:
        query (str): 검색 키워드 또는 질문

    Returns:
        str: 검색 결과 요약
    """
    return search_engine.search(query=query, max_results=3)


@mcp.tool()
def list_google_calendar_events(max_results: int = 10) -> str:
    """Google Calendar에서 향후 예정된 일정 목록을 조회합니다."""
    try:
        events = calendar_adapter.list_events(max_results=max_results)
        if not events:
            return "예정된 Google Calendar 일정이 없습니다."

        results = []
        for idx, event in enumerate(events, 1):
            results.append(
                f"[{idx}] {event['title']}\n"
                f"   - 일시: {event['start']} ~ {event['end']}\n"
                f"   - 장소: {event['location'] or '없음'}\n"
                f"   - 메모: {event['description'] or '없음'}"
            )
        return "\n\n".join(results)
    except Exception as e:
        return f"Google Calendar 일정 조회 중 오류 발생: {str(e)}"

@mcp.tool()
def search_financial_guide(query: str) -> str:
    """금융투자협회 투자길라잡이(2018) 문서 VectorDB에서 관련 지식 및 투자 정보를 검색합니다.

    Args:
        query (str): 금융, 주식, 펀드, 투자 관련 질문 또는 검색 키워드

    Returns:
        str: 검색된 문서 조각 및 관련 정보
    """
    return vector_db_adapter.search_similar_documents(query=query, k=3)


# ==========================================
#  MCP Resources (자원)
# ==========================================

@mcp.resource("system://info")
def get_system_info() -> str:
    """현재 MCP 서버가 동작하는 시스템 및 OS 환경 정보 리소스"""
    return f"""[MCP Server System Status]
- OS: {platform.system()} {platform.release()} ({platform.architecture()[0]})
- Python Version: {platform.python_version()}
- Current Server Time: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- Active Services: Safe Calculator, Open-Meteo Weather, Tavily Web Search, Financial Guide FAISS VectorDB, Google Calendar
"""


@mcp.resource("config://app-guide")
def get_app_guide() -> str:
    """MCP 시스템 이용 가이드 리소스"""
    return """[MCP Application Guide]
1. 수학 연산: AST Sandbox 기반으로 안전하게 사칙연산을 수행합니다.
2. 날씨 정보 조회: 'get_weather_forecast' 툴로 실시간 날씨 및 예보를 확인합니다.
3. 실시간 정보 검색: TavilySearchEngine을 통해 최신 웹 정보를 조회합니다.
4. 금융 정보 조회: 'search_financial_guide' 툴을 활용하여 FAISS VectorDB에서 검색합니다.
5. 일정 관리: 'create_google_calendar_event', 'list_google_calendar_events' 툴로 구글 캘린더를 연동합니다.
"""




if __name__ == "__main__":
    mcp.run(transport="stdio")