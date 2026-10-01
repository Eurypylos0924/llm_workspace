import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

load_dotenv(override=True)


ev_technology_agent = LlmAgent(
    name="ev_technology_agent",

    model=LiteLlm(
        model="openai/z-ai/glm-5.3-flash",
        api_base="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("nvidiaapi_key"),
    ),

    description="전기차 기술을 연구하는 에이전트",

    instruction="""
당신은 교통 분야 AI 연구 보조입니다.

'전기차 기술'의 최근 발전사항을 조사하세요.

핵심 발견 내용을 간결하게 1~2문장으로 요약하고,
요약만 출력하세요.
""",

    output_key="ev_technology_result",
)