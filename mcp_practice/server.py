# ============================================================
# [MCP SERVER 영역]
# 노트북에서 실행하지 않는다. 클라이언트가 별도 프로세스로 실행한다.
# (mcp 2.x: FastMCP -> MCPServer 로 이름 변경)
# ============================================================
from mcp.server.mcpserver import MCPServer

mcp = MCPServer("Weather Server")


# @mcp.tool(): 아래 함수를 MCP Tool로 등록 (이름, docstring, 타입 힌트로 스키마 생성)
@mcp.tool()
def get_weather(city: str) -> str:
    """도시의 날씨를 조회한다."""
    return f"{city}의 날씨는 맑음, 20도"


@mcp.tool()
def multiply(a: int, b: int) -> int:
    """두 정수를 곱한다."""
    return a * b


if __name__ == "__main__":
    mcp.run()  # 기본 transport = stdio
