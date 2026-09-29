# ============================================================
# [MCP CLIENT 영역] 터미널 실행용
#   python mcp_practice/client.py
# ============================================================
import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

SERVER_PATH = Path(__file__).parent / "server.py"

# 어떤 서버를 어떻게 실행할지: 현재 파이썬으로 server.py 실행
server_params = StdioServerParameters(command=sys.executable, args=[str(SERVER_PATH)])


async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            # MCP Protocol: Tool 목록 요청
            tools = await session.list_tools()
            print("사용 가능한 MCP Tool")
            for tool in tools.tools:
                print(f"- {tool.name}: {tool.description}")
                print(f"  input_schema: {tool.input_schema}")

            # MCP Protocol: Tool 호출
            result = await session.call_tool("get_weather", {"city": "서울"})
            print("\nget_weather 결과:", result.content[0].text)

            result = await session.call_tool("multiply", {"a": 23, "b": 45})
            print("multiply 결과:", result.content[0].text)


if __name__ == "__main__":
    asyncio.run(main())
