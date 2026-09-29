
# app/api/routes.py (REST API 라우터)
# 웹 프론트엔드 및 외부 REST 클라이언트와 통신할 API 엔드포인트를 정의합니다.
import traceback
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from app.services.mcp_service import MCPService

router = APIRouter(prefix="/api", tags=["MCP Operations"])
mcp_service = MCPService()


class ChatRequest(BaseModel):
    prompt: str


class ChatResponse(BaseModel):
    answer: str
    tool_used: str | None = None
    tool_result: str | None = None


# http://127.0.0.1:8000/api/tools
@router.get("/tools")
async def get_tools():
    """현재 MCP Server에 등록되어 있는 도구 목록을 반환합니다."""
    try:
        tools = await mcp_service.get_available_tools()
        return {"tools": tools}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"MCP Tools 조회 실패: {str(e)}"
        )


# http://127.0.0.1:8000/api/chat
@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        response = await mcp_service.process_chat(request.prompt)
        return response
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"MCP 연산 처리 중 에러: {str(e)}"
        )




# http://127.0.0.1:8000/api/resource/system-info
@router.get("/resource/system-info")
async def get_system_info():
    """system://info MCP Resource를 실행하여 반환하는 엔드포인트"""
    try:
        info_text = await mcp_service.get_system_info_resource()
        return {"resource_uri": "system://info", "content": info_text}
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Resource 읽기 실패: {str(e)}"
        )




# http://127.0.0.1:8000/api/resource/app-guide
@router.get("/resource/app-guide")
async def get_app_guide():
    """config://app-guide MCP Resource를 읽어서 반환하는 REST API"""
    try:
        guide_text = await mcp_service.get_app_guide_resource()
        return {
            "resource_uri": "config://app-guide",
            "content": guide_text
        }
    except Exception as e:
        traceback.print_exc()
        raise HTTPException(
            status_code=500, detail=f"Resource 읽기 실패: {str(e)}"
        )