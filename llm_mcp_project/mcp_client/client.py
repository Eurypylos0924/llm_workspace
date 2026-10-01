# mcp_client/client.py

import asyncio
import os
import sys
from typing import Any, Dict, List

import anyio
from dotenv import load_dotenv

load_dotenv(override=True)

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


class MCPClient:

    def __init__(self, model_name: str = os.getenv("LLM_MODEL", "z-ai/glm-5.3-flash")):
        # NVIDIA OpenAI 호환 엔드포인트
        self.llm = ChatOpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.getenv("nvidiaapi_key"),
            model=model_name,
            temperature=0,
        )
        self.llm_with_tools = None
        self.session: ClientSession | None = None
        self.available_tools: List[Dict[str, Any]] = []

    async def connect_and_run(self, user_prompt: str):
        """MCP Server에 연결하고 백그라운드 작업 그룹(Task Group) 내에서 session을 유지 및 실행합니다."""
        env = os.environ.copy()
        env["PYTHONPATH"] = os.getcwd()

        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", "mcp_server.server"],
            env=env,
        )

        # 1. stdio 연결 생성
        async with stdio_client(server_params) as (read, write):
            # 2. ClientSession 인스턴스 생성
            async with ClientSession(read, write) as session:
                self.session = session
                
                # 4. 세션 초기화 
                await session.initialize()
                print("✅ [MCP Host] MCP Server 세션 초기화 완료!")

                # 5. Tool 목록 가져오기 및 LangChain 바인딩
                tools_response = await session.list_tools()
                self.available_tools = [
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
                self.llm_with_tools = self.llm.bind_tools(
                    self.available_tools
                )

                # 6. 프롬프트 실행
                result = await self.run_prompt(user_prompt)
                print(f"\n💬 [최종 답변]:\n{result}")

                    

    async def run_prompt(self, user_prompt: str) -> str:
        if not self.session or not self.llm_with_tools:
            raise RuntimeError("MCP Server 세션이 활성화되지 않았습니다.")

        messages = [HumanMessage(content=user_prompt)]
        ai_msg: AIMessage = await self.llm_with_tools.ainvoke(messages)
        messages.append(ai_msg)

        if ai_msg.tool_calls:
            for tool_call in ai_msg.tool_calls:
                function_name = tool_call["name"]
                function_args = tool_call["args"]
                tool_call_id = tool_call["id"]

                print(f"🤖 [LangChain] MCP Tool '{function_name}' 호출 중...")

                # session.call_tool 호출
                tool_result = await self.session.call_tool(
                    name=function_name, arguments=function_args
                )

                result_text = "".join(
                    [
                        content.text
                        for content in tool_result.content
                        if content.type == "text"
                    ]
                )
                print(f"⚙️ [MCP Tool 결과]: {result_text}")

                messages.append(
                    ToolMessage(
                        content=result_text, tool_call_id=tool_call_id
                    )
                )

            final_ai_msg: AIMessage = await self.llm.ainvoke(messages)
            return str(final_ai_msg.content)
        else:
            return str(ai_msg.content)


async def main():
    host = MCPClient()
    # 연결 및 실행
    await host.connect_and_run("12.5에 4를 곱한 결과를 알려줘.")


if __name__ == "__main__":
   asyncio.run(main())