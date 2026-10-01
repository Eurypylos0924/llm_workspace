# app/services/mcp_service.py

"""
MCP Service

역할
1. MCP Server를 Stdio 방식으로 실행한다.
2. MCP Server에 등록된 Tool 목록을 조회한다.
3. LLM에게 MCP Tool을 전달하고 Tool Calling을 처리한다.
4. MCP Tool을 실제 MCP Server에서 실행한다.
5. MCP Resource를 조회한다.
6. FastAPI REST API에서 사용할 수 있는 형태로 결과를 반환한다.
"""

import datetime
import os
import sys
import traceback

from typing import Any, Dict, List

from dotenv import load_dotenv

from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_openai import ChatOpenAI

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from app.core.settings import PROJECT_ROOT, get_settings


# ============================================================
# 1. 환경 변수 로드
# ============================================================

load_dotenv(override=True)

settings = get_settings()


# ============================================================
# 2. MCPService
# ============================================================

class MCPService:
    """
    FastAPI와 MCP Server 사이를 연결하는 서비스 클래스이다.

    전체 구조

    FastAPI
       ↓
    MCPService
       ↓
    MCP Client
       ↓
    MCP Protocol
       ↓
    MCP Server
       ↓
    @mcp.tool()
    @mcp.resource()
    """

    def __init__(self, model_name: str = settings.llm_model):

        # --------------------------------------------------------
        # NVIDIA API Key 확인
        # --------------------------------------------------------

        api_key = (
            os.getenv("nvidiaapi_key")
            or settings.nvidiaapi_key
        )

        if not api_key:
            raise ValueError(
                "nvidiaapi_key가 설정되어 있지 않습니다. "
                ".env 파일이나 환경변수를 확인해 주세요."
            )

        # --------------------------------------------------------
        # LLM 생성 (NVIDIA OpenAI 호환 엔드포인트)
        # --------------------------------------------------------

        self.llm = ChatOpenAI(
            base_url=settings.nvidia_base_url,
            model=model_name,
            temperature=0,
            api_key=api_key,
        )

    # ============================================================
    # 3. MCP Server 실행 파라미터
    # ============================================================

    def _get_server_params(self) -> StdioServerParameters:
        """
        MCP Server를 Stdio 방식으로 실행하기 위한
        서버 실행 파라미터를 생성한다.
        """

        # 현재 환경 변수 복사
        env = os.environ.copy()

        # 프로젝트 루트를 Python Module 검색 경로에 추가
        env["PYTHONPATH"] = str(PROJECT_ROOT)

        # Windows UTF-8 문제 방지
        env["PYTHONIOENCODING"] = "utf-8"
        env["PYTHONUTF8"] = "1"

        # --------------------------------------------------------
        # MCP Server 실행
        #
        # python -m mcp_server.server
        # --------------------------------------------------------

        return StdioServerParameters(
            command=sys.executable,
            args=[
                "-m",
                "mcp_server.server",
            ],
            env=env,
        )

    # ============================================================
    # 4. MCP Server Tool 목록 조회
    # ============================================================

    async def get_available_tools(self) -> List[Dict[str, Any]]:
        """
        MCP Server에 등록되어 있는 Tool 목록을 조회한다.
        """

        try:

            server_params = self._get_server_params()

            # ----------------------------------------------------
            # MCP Server 실행
            # ----------------------------------------------------

            async with stdio_client(server_params) as (read, write):

                # MCP Client Session 생성
                async with ClientSession(read, write) as session:

                    # MCP 초기화 / Handshake
                    await session.initialize()

                    # ------------------------------------------------
                    # MCP Server Tool 목록 조회
                    # ------------------------------------------------

                    tools_response = await session.list_tools()

                    # ------------------------------------------------
                    # REST API에서 사용하기 편한 Dictionary로 변환
                    # ------------------------------------------------

                    return [
                        {
                            "name": tool.name,
                            "description": tool.description,
                            "input_schema": tool.input_schema,
                        }
                        for tool in tools_response.tools
                    ]

        except Exception as e:

            print(
                "❌ [MCPService.get_available_tools 에러]"
            )

            traceback.print_exc()

            raise e

    # ============================================================
    # 5. MCP Resource 조회
    # ============================================================

    async def read_mcp_resource(
        self,
        resource_uri: str,
    ) -> str:
        """
        MCP Resource URI를 받아 Resource 데이터를 조회한다.

        예:

        system://info
        config://app-guide
        """

        try:

            # ----------------------------------------------------
            # MCP Server 실행 파라미터
            # ----------------------------------------------------

            server_params = self._get_server_params()

            # ----------------------------------------------------
            # MCP Server와 Stdio 연결
            # ----------------------------------------------------

            async with stdio_client(server_params) as (read, write):

                # MCP Session 생성
                async with ClientSession(read, write) as session:

                    # MCP Handshake
                    await session.initialize()

                    # ------------------------------------------------
                    # Resource 조회
                    # ------------------------------------------------

                    resource_result = await session.read_resource(
                        resource_uri
                    )

                    # ------------------------------------------------
                    # Resource 결과에서 text만 추출
                    # ------------------------------------------------

                    texts = []

                    for content in resource_result.contents:

                        if hasattr(content, "text"):

                            texts.append(content.text)

                    return "\n".join(texts)

        except Exception as e:

            print(
                f"❌ [MCPService.read_mcp_resource "
                f"({resource_uri}) 에러]"
            )

            traceback.print_exc()

            raise e

    # ============================================================
    # 6. System Resource
    # ============================================================

    async def get_system_info_resource(self) -> str:
        """
        system://info MCP Resource를 조회한다.
        """

        return await self.read_mcp_resource(
            "system://info"
        )

    # ============================================================
    # 7. App Guide Resource
    # ============================================================

    async def get_app_guide_resource(self) -> str:
        """
        config://app-guide MCP Resource를 조회한다.
        """

        return await self.read_mcp_resource(
            "config://app-guide"
        )

    # ============================================================
    # 8. 현재 날짜/시간 생성
    # ============================================================

    def _get_current_time_string(self) -> str:
        """
        LLM이 '오늘', '내일', '모레' 등의 상대 날짜를
        처리할 수 있도록 현재 날짜/시간을 문자열로 생성한다.
        """

        now = datetime.datetime.now()

        days_ko = [
            "월요일",
            "화요일",
            "수요일",
            "목요일",
            "금요일",
            "토요일",
            "일요일",
        ]

        return (
            f"{now.strftime('%Y년 %m월 %d일')} "
            f"{days_ko[now.weekday()]} "
            f"{now.strftime('%H시 %M분')}"
        )

    # ============================================================
    # 9. LLM System Prompt
    # ============================================================

    def _create_system_prompt(self) -> str:
        """
        MCP Tool Calling을 위한 시스템 프롬프트를 생성한다.
        """

        current_time_str = self._get_current_time_string()

        return f"""
당신은 MCP 도구를 활용하여 사용자의 요청을 처리하는 AI 비서입니다.

[현재 시각 정보]
오늘 날짜/시간:
{current_time_str}

[Tool 사용 규칙]

1. 사용자의 요청을 분석하여 MCP Tool이 필요한 경우
   적절한 Tool을 선택하여 호출하세요.

2. Tool의 입력값은 Tool의 input schema에 맞게 작성하세요.

3. 계산, 데이터 조회, 외부 시스템 조회 등
   Tool이 필요한 작업은 가능한 경우 MCP Tool을 사용하세요.

4. Tool 실행 결과를 받은 후 사용자가 이해하기 쉬운
   자연어로 최종 답변을 작성하세요.

5. 사용자가 '내일', '모레', '다음주' 등의 상대적인
   날짜를 사용하는 경우 현재 시각을 기준으로 정확한
   날짜를 계산하세요.

6. Google Calendar 관련 Tool을 호출하는 경우
   start_time과 end_time은 ISO 8601 형식을 사용하세요.

예:

2026-09-29T10:00:00
"""

    # ============================================================
    # 10. 자연어 → MCP Tool 실행
    # ============================================================

    async def process_chat(
        self,
        user_prompt: str,
    ) -> Dict[str, Any]:
        """
        자연어 프롬프트를 받아

        사용자
          ↓
        LLM
          ↓
        Tool Calling
          ↓
        MCP Client
          ↓
        MCP Server
          ↓
        @mcp.tool()

        구조로 Tool을 실행한다.

        Resource 요청은 별도의 Resource 경로로 처리한다.
        """

        try:

            # ----------------------------------------------------
            # 입력값 정리
            # ----------------------------------------------------

            user_prompt = user_prompt.strip()

            if not user_prompt:

                return {
                    "answer": "프롬프트를 입력해주세요.",
                    "tool_used": None,
                    "tool_result": None,
                    "resource_used": None,
                }

            # ====================================================
            # A. Resource 직접 요청
            # ====================================================

            if (
                "시스템 정보" in user_prompt
                or "서버 상태" in user_prompt
            ):

                info_text = (
                    await self.get_system_info_resource()
                )

                # ----------------------------------------------
                # 중요:
                # routes.py의 ChatResponse와 동일한 구조
                # ----------------------------------------------

                return {
                    "answer": info_text,
                    "tool_used": None,
                    "tool_result": None,
                    "resource_used": "system://info",
                }

            # ====================================================
            # B. App Guide Resource 요청
            # ====================================================

            if (
                "사용법" in user_prompt
                or "앱 사용법" in user_prompt
                or "도움말" in user_prompt
                 or "가이드" in user_prompt
            ):

                guide_text = (
                    await self.get_app_guide_resource()
                )

                return {
                    "answer": guide_text,
                    "tool_used": None,
                    "tool_result": None,
                    "resource_used": "config://app-guide",
                }

            # ====================================================
            # C. 일반 MCP Tool 처리
            # ====================================================

            server_params = self._get_server_params()

            # ----------------------------------------------------
            # MCP Server 연결
            # ----------------------------------------------------

            async with stdio_client(server_params) as (
                read,
                write,
            ):

                async with ClientSession(
                    read,
                    write,
                ) as session:

                    # ------------------------------------------------
                    # MCP 초기화
                    # ------------------------------------------------

                    await session.initialize()

                    # ------------------------------------------------
                    # MCP Server Tool 목록 조회
                    # ------------------------------------------------

                    tools_response = (
                        await session.list_tools()
                    )

                    # =================================================
                    # MCP Tool
                    #        ↓
                    # OpenAI Function Calling Schema
                    # =================================================

                    openai_tools = [

                        {
                            "type": "function",

                            "function": {

                                "name": tool.name,

                                "description": (
                                    tool.description
                                    or ""
                                ),

                                "parameters": (
                                    tool.input_schema
                                ),
                            },
                        }

                        for tool
                        in tools_response.tools
                    ]

                    # ------------------------------------------------
                    # System Prompt
                    # ------------------------------------------------

                    system_prompt = (
                        self._create_system_prompt()
                    )

                    # ------------------------------------------------
                    # LLM + MCP Tools 연결
                    # ------------------------------------------------

                    llm_with_tools = (
                        self.llm.bind_tools(
                            openai_tools
                        )
                    )

                    # ------------------------------------------------
                    # 대화 메시지 생성
                    # ------------------------------------------------

                    messages = [

                        SystemMessage(
                            content=system_prompt
                        ),

                        HumanMessage(
                            content=user_prompt
                        ),
                    ]

                    # =================================================
                    # LLM 1차 호출
                    # =================================================

                    ai_msg: AIMessage = (
                        await llm_with_tools.ainvoke(
                            messages
                        )
                    )

                    # AI 메시지를 대화 기록에 추가
                    messages.append(ai_msg)

                    # ------------------------------------------------
                    # Tool 실행 정보 초기화
                    # ------------------------------------------------

                    tool_used = None
                    tool_result_text = None

                    # =================================================
                    # D. LLM이 Tool을 선택했는지 확인
                    # =================================================

                    if ai_msg.tool_calls:

                        # 여러 개의 Tool 호출도 처리
                        for tool_call in ai_msg.tool_calls:

                            # ----------------------------------------
                            # Tool 이름
                            # ----------------------------------------

                            function_name = (
                                tool_call["name"]
                            )

                            # ----------------------------------------
                            # Tool 입력값
                            # ----------------------------------------

                            function_args = (
                                tool_call["args"]
                            )

                            # ----------------------------------------
                            # Tool Call ID
                            # ----------------------------------------

                            tool_call_id = (
                                tool_call["id"]
                            )

                            # ----------------------------------------
                            # Tool 이름 기록
                            # ----------------------------------------

                            tool_used = function_name

                            print(
                                f"🔧 MCP Tool 호출: "
                                f"{function_name}"
                            )

                            print(
                                f"📥 Tool arguments: "
                                f"{function_args}"
                            )

                            # =================================================
                            # MCP Server Tool 실행
                            # =================================================

                            result = (
                                await session.call_tool(
                                    name=function_name,
                                    arguments=function_args,
                                )
                            )

                            # -------------------------------------------------
                            # Tool 결과에서 text 추출
                            # -------------------------------------------------

                            result_texts = []

                            for content in result.content:

                                if (
                                    content.type
                                    == "text"
                                ):

                                    result_texts.append(
                                        content.text
                                    )

                            tool_result_text = (
                                "\n".join(
                                    result_texts
                                )
                            )

                            print(
                                f"📤 Tool result: "
                                f"{tool_result_text}"
                            )

                            # =================================================
                            # ToolMessage 생성
                            # =================================================

                            messages.append(
                                ToolMessage(
                                    content=(
                                        tool_result_text
                                    ),
                                    tool_call_id=(
                                        tool_call_id
                                    ),
                                )
                            )

                        # =================================================
                        # E. Tool 결과를 다시 LLM에게 전달
                        # =================================================

                        final_ai_msg: AIMessage = (
                            await self.llm.ainvoke(
                                messages
                            )
                        )

                        final_answer = (
                            str(
                                final_ai_msg.content
                            )
                        )

                    else:

                        # =================================================
                        # Tool이 필요 없는 일반 질문
                        # =================================================

                        final_answer = (
                            str(
                                ai_msg.content
                            )
                        )

                    # =================================================
                    # F. 최종 응답
                    # =================================================

                    return {

                        "answer": final_answer,

                        "tool_used": tool_used,

                        "tool_result": (
                            tool_result_text
                        ),

                        "resource_used": None,
                    }

        except Exception as e:

            print(
                "❌ [MCPService.process_chat 에러]"
            )

            traceback.print_exc()

            raise e