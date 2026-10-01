import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
from ...tools.web_search import tavily_web_search  # 상대 경로: 어느 폴더에서 adk web을 실행해도 동작

load_dotenv(override=True)

renewable_energy_agent = LlmAgent(
    name="renewable_energy_agent",

    model=LiteLlm(
        model="openai/z-ai/glm-5.3-flash",
        api_base="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("nvidiaapi_key"),
    ),

    description="재생 가능 에너지 소스 연구 에이전트",

    instruction="""
당신은 재생 가능 에너지 분야의 AI 연구 보조입니다.

'재생 에너지 소스'의 최신 발전 동향을 조사하세요.

핵심 발견 내용을 간결하게 1~2문장으로 요약하고,
요약만 출력하세요.
""",

   tools=[
        tavily_web_search
    ],

    output_key="renewable_energy_result",
)