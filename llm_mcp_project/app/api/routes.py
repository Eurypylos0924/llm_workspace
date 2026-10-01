# app/api/routes.py

import traceback

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.services.mcp_service import MCPService


# ============================================================
# Router 설정
# ============================================================

router = APIRouter(
    prefix="/api",
    tags=["MCP Operations"]
)


# MCP Service 객체 생성
mcp_service = MCPService()


# ============================================================
# Request / Response Model
# ============================================================

class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    answer: str
    tool_used: str | None = None
    tool_result: str | None = None
    resource_used: str | None = None


# ============================================================
# MCP Tool 목록 조회
# ============================================================

# GET
# http://127.0.0.1:8000/api/tools

@router.get("/tools")
async def get_tools():

    """현재 MCP Server에 등록된 Tool 목록을 반환한다."""

    try:

        tools = await mcp_service.get_available_tools()

        return {
            "tools": tools
        }

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"MCP Tools 조회 실패: {str(e)}"
        )


# ============================================================
# 자연어 MCP Chat
# ============================================================

# POST
# http://127.0.0.1:8000/api/chat

@router.post(
    "/chat",
    response_model=ChatResponse
)
async def chat_endpoint(request: ChatRequest):

    try:

        # 사용자 입력 제거
        user_prompt = request.prompt.strip()

        # 빈 입력 방지
        if not user_prompt:

            raise HTTPException(
                status_code=400,
                detail="프롬프트를 입력해주세요."
            )

        # MCP Service 실행
        response = await mcp_service.process_chat(
            user_prompt
        )

        return response

    except HTTPException:
        raise

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"MCP 연산 처리 중 에러: {str(e)}"
        )


# ============================================================
# system://info Resource
# ============================================================

# GET
# http://127.0.0.1:8000/api/resource/system-info

@router.get("/resource/system-info")
async def get_system_info():

    """
    system://info MCP Resource를 읽어서 반환한다.
    """

    try:

        info_text = (
            await mcp_service.get_system_info_resource()
        )

        return {
            "resource_uri": "system://info",
            "content": info_text
        }

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Resource 읽기 실패: {str(e)}"
        )


# ============================================================
# config://app-guide Resource
# ============================================================

# GET
# http://127.0.0.1:8000/api/resource/app-guide

@router.get("/resource/app-guide")
async def get_app_guide():

    """
    config://app-guide MCP Resource를 읽어서 반환한다.
    """

    try:

        guide_text = (
            await mcp_service.get_app_guide_resource()
        )

        return {
            "resource_uri": "config://app-guide",
            "content": guide_text
        }

    except Exception as e:

        traceback.print_exc()

        raise HTTPException(
            status_code=500,
            detail=f"Resource 읽기 실패: {str(e)}"
        )