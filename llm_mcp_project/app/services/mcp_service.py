import datetime
import os
import sys
import traceback
from typing import Any, Dict, List

from dotenv import load_dotenv

# .env 파일에 등록된 환경 변수들을 읽어오며, 이미 등록된 환경 변수가 있더라도 덮어씁니다 (override=True).
load_dotenv(override=True)

# LangChain 메시지 규격 클래스들을 임포트합니다.
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
# OpenAI 호환 Chat 모델 연동 클래스 (NVIDIA 엔드포인트에 base_url로 연결)를 임포트합니다.
from langchain_openai import ChatOpenAI
# Model Context Protocol (MCP) 클라이언트 세션 및 Stdio 서버 연결 파라미터 클래스를 임포트합니다.
from mcp import ClientSession, StdioServerParameters
# 프로세스 간 표준 입출력(Stdio) 방식으로 MCP 서버와 통신하는 클라이언트 함수를 임포트합니다.
from mcp.client.stdio import stdio_client

# 프로젝트 루트 경로 설정 및 통합 설정 객체 로더를 임포트합니다.
from app.core.settings import PROJECT_ROOT, get_settings
# 로컬에서 직접 연산 및 실행을 진행하기 위한 외부 어댑터 모듈들을 임포트합니다.
from external_adapter.python_sandbox import SafePythonSandbox
from external_adapter.tavily_search import TavilySearchEngine
from external_adapter.vector_db import VectorDBAdapter
from external_adapter.weather import WeatherAdapter

# 애플리케이션의 공통 설정 객체를 생성합니다.
settings = get_settings()


class MCPService:
    """FastAPI 내부에서 MCP Server 실행 및 LangChain LLM과의 중계를 담당하는 서비스 클래스"""

    def __init__(self, model_name: str = settings.llm_model):
        # OS 환경 변수에서 NVIDIA API 키를 먼저 조회하고, 없을 경우 settings 객체에서 가져옵니다.
        api_key = os.getenv("nvidiaapi_key") or settings.nvidiaapi_key

        # API 키가 존재하지 않을 경우 명시적으로 예외(ValueError)를 발생시킵니다.
        if not api_key:
            raise ValueError(
                "nvidiaapi_key가 설정되어 있지 않습니다. .env 파일이나 환경변수를 확인해 주세요."
            )

        # NVIDIA는 OpenAI 호환 API를 제공하므로 ChatOpenAI에 base_url만 NVIDIA 엔드포인트로 지정합니다.
        self.llm = ChatOpenAI(
            base_url=settings.nvidia_base_url,
            model=model_name,
            temperature=0,
            api_key=api_key,
        )

        # 안전한 파이썬 코드 실행을 위한 샌드박스 어댑터 인스턴스를 생성합니다.
        self.sandbox = SafePythonSandbox()

        # 날씨 정보 조회를 위한 어댑터 인스턴스를 생성합니다.
        self.weather_adapter = WeatherAdapter()

        # Tavily 검색 엔진 조회를 위한 어댑터 인스턴스를 생성합니다.
        self.search_engine = TavilySearchEngine()

        # 벡터 데이터베이스 연동 및 RAG 조회를 위한 어댑터 인스턴스를 생성합니다 (API 키 전달).
        self.vector_db_adapter = VectorDBAdapter(nvidia_api_key=api_key)

    def _get_server_params(self) -> StdioServerParameters:
        """MCP Server 구동을 위한 StdioServerParameters 생성 헬퍼 함수"""
        # 현재 시스템의 환경 변수 딕셔너리를 복사합니다.
        env = os.environ.copy()

        # MCP 서버 프로세스가 프로젝트 루트 파이썬 모듈을 정상적으로 참조할 수 있도록 PYTHONPATH를 설정합니다.
        env["PYTHONPATH"] = str(PROJECT_ROOT)

        # Windows 환경 등에서 발생할 수 있는 입출력 인코딩 오류를 방지하기 위해 UTF-8 설정을 강제 적용합니다.
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        # 파이썬 실행 파이프라인 명령어 및 모듈 실행 인자(-m mcp_server.server)와 환경 변수를 세팅하여 반환합니다.
        return StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            env=env,
        )

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """MCP Server에 등록된 사용 가능한 툴 목록 조회"""
        try:
            # MCP 서버 구동에 필요한 설정 파라미터를 생성합니다.
            server_params = self._get_server_params()

            # Stdio 클라이언트를 생성하여 MCP 서버와 비동기 세션을 엽니다.
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # MCP 서버 세션을 초기화 핸드셰이크합니다.
                    await session.initialize()

                    # 서버 측에 등록된 전체 툴 목록을 비동기 조회합니다.
                    tools_response = await session.list_tools()

                    # MCP Tool 객체 구조를 딕셔너리 형태로 정제하여 리스트로 반환합니다.
                    return [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.input_schema,
                        }
                        for tool in tools_response.tools
                    ]
        except Exception as e:
            # 에러 발생 시 로그 출력 및 스택 트레이스를 기록한 뒤 예외를 상위로 다시 던집니다.
            print("❌ [MCPService.get_available_tools 에러]:")
            traceback.print_exc()
            raise e

    async def process_chat(self, user_prompt: str) -> Dict[str, Any]:
        """자연어 프롬프트를 받아 MCP Tool을 자동 선택하여 연산 후 결과 반환"""
        try:
            # MCP 서버 통신을 위한 실행 파라미터를 가져옵니다.
            server_params = self._get_server_params()

            # MCP 서버와 연결되는 Stdio 세션을 연결합니다.
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # 세션 초기화를 진행합니다.
                    await session.initialize()

                    # 1. MCP Server로부터 지원 가능한 툴 목록을 조회합니다.
                    tools_response = await session.list_tools()

                    # 조회된 MCP 툴들을 OpenAI Function Calling 스키마에 맞춰 변환합니다.
                    openai_tools = [
                        {
                            "type": "function",
                            "function": {
                                "name": tool.name,
                                "description": tool.description,
                                "parameters": tool.input_schema,
                            },
                        }
                        for tool in tools_response.tools
                    ]

                    # 현재 시각을 구하여 상대적 날짜/시간 추론 기준점으로 사용합니다.
                    now = datetime.datetime.now()

                    # 요일 변환을 위한 한국어 이름 리스트입니다.
                    days_ko = ["월요일", "화요일", "수요일", "목요일", "금요일", "토요일", "일요일"]

                    # 가독성 있는 형태의 현재 날짜/시각 문자열을 생성합니다.
                    current_time_str = f"{now.strftime('%Y년 %m월 %d일')} {days_ko[now.weekday()]} {now.strftime('%H시 %M분')}"

                    # 모델에게 현재 시각 정보 및 캘린더 도구 사용 규칙 지침을 명시하는 시스템 프롬프트를 구성합니다.
                    system_prompt = (
                        f"당신은 MCP 도구를 활용하여 사용자의 요청을 처리하는 AI 비서입니다.\n\n"
                        f"[현재 시각 정보]\n"
                        f"- 오늘 날짜/시간: {current_time_str}\n\n"
                        f"[지침]\n"
                        f"1. 사용자가 '내일', '모레', '다음주' 등 상대적인 날짜를 언급하면 반드시 위 **[현재 시각 정보]**를 기준점으로 삼아 정확한 날짜(YYYY-MM-DD)를 계산하세요.\n"
                        f"2. Google Calendar 관련 도구 호출 시 `start_time`과 `end_time`은 ISO 8601 형식(예: '2026-09-29T10:00:00')으로 작성해 호출하세요.\n"
                    )

                    # 2. ChatOpenAI 모델에 바인딩할 도구 목록을 설정합니다.
                    llm_with_tools = self.llm.bind_tools(openai_tools)

                    # 시스템 프롬프트와 사용자 입력 프롬프트를 담은 메시지 리스트를 생성합니다.
                    messages = [
                        SystemMessage(content=system_prompt),
                        HumanMessage(content=user_prompt)
                    ]
                    
                    # LLM을 호출하여 도구 사용 여부 판단 및 1차 추론 응답 메시지를 받습니다.
                    ai_msg: AIMessage = await llm_with_tools.ainvoke(messages)

                    # 생성된 AI 메시지를 대화 기록 리스트에 추가합니다.
                    messages.append(ai_msg)

                    # 실행된 도구 명칭 및 실행 결과를 저장하기 위한 변수를 초기화합니다.
                    tool_used = None
                    tool_result_text = None

                    # 3. 모델이 하나 이상의 Tool 호출을 판단했는지 확인합니다.
                    if ai_msg.tool_calls:
                        # 요청된 도구 호출 목록을 순회하며 실행합니다.
                        for tool_call in ai_msg.tool_calls:
                            # 호출할 함수(도구)의 이름, 전달 인자, 고유 ID를 추출합니다.
                            function_name = tool_call["name"]
                            function_args = tool_call["args"]
                            tool_call_id = tool_call["id"]

                            # 최근 사용된 도구의 이름을 보관합니다.
                            tool_used = function_name

                            # MCP Server에 요청하여 해당 도구를 실제로 실행하고 결과를 받습니다.
                            result = await session.call_tool(
                                name=function_name, arguments=function_args
                            )

                            # 실행 결과 내 텍스트 타입의 컨텐츠들을 하나의 문자열로 결합합니다.
                            tool_result_text = "".join(
                                [
                                    content.text
                                    for content in result.content
                                    if content.type == "text"
                                ]
                            )

                            # 도구 수행 결과를 ToolMessage 형태로 변환하여 대화 기록 목록에 추가합니다.
                            messages.append(
                                ToolMessage(
                                    content=tool_result_text,
                                    tool_call_id=tool_call_id,
                                )
                            )

                        # 4. 도구 실행 결과를 바탕으로 최종 자연어 답변을 얻기 위해 LLM을 재호출합니다.
                        final_ai_msg: AIMessage = await self.llm.ainvoke(messages)
                        final_answer = str(final_ai_msg.content)
                    else:
                        # 도구 호출이 필요 없는 일반 질문인 경우 1차 응답 텍스트를 그대로 최종 답변으로 사용합니다.
                        final_answer = str(ai_msg.content)

                    # 최종 답변, 사용된 도구명, 도구 실행 결과 문자열을 반환합니다.
                    return {
                        "answer": final_answer,
                        "tool_used": tool_used,
                        "tool_result": tool_result_text,
                    }
        except Exception as e:
            # 예외 발생 시 트레이스백 로그를 출력하고 상위로 예외를 전파합니다.
            print("❌ [MCPService.process_chat 에러]:")
            traceback.print_exc()
            raise e

    async def read_mcp_resource(self, resource_uri: str) -> str:
        """전달받은 URI 문자열의 MCP Resource 데이터를 조회하여 반환"""
        try:
            # MCP 서버 구동을 위한 파라미터를 준비합니다.
            server_params = self._get_server_params()

            # Stdio 클라이언트를 구동하여 MCP 서버 세션을 엽니다.
            async with stdio_client(server_params) as (read, write):
                async with ClientSession(read, write) as session:
                    # 세션 핸드셰이크를 진행합니다.
                    await session.initialize()

                    # 서버 측으로부터 리소스 URI에 해당하는 리소스 데이터를 비동기로 읽어옵니다.
                    resource_result = await session.read_resource(resource_uri)

                    # 텍스트 속성을 보유한 컨텐츠 항목만 추출해 줄바꿈 문자('\n')로 합쳐 반환합니다.
                    return "\n".join(
                        [
                            content.text
                            for content in resource_result.contents
                            if hasattr(content, "text")
                        ]
                    )
        except Exception as e:
            # 리소스 조회 실패 시 해당 URI 명시와 함께 트레이스백을 출력하고 예외를 전파합니다.
            print(f"❌ [MCPService.read_mcp_resource ({resource_uri}) 에러]:")
            traceback.print_exc()
            raise e