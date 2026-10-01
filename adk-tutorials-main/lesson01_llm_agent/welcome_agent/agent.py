import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

load_dotenv(override=True)

# 현재 에이전트는 하나이므로 이 에이전트가 root_agent이므로 이름을 반드시 root_agent로 해야한다.
# instruction: 프롬프트에 해당한다.

# ADK는 Gemini 중심으로 동작하며, OpenAI와 같은 비-Gemini 모델은 현재 공식 패턴상 LiteLlm을 통해 연결하는 방식이 사용된다.
root_agent = LlmAgent(
    name="welcome_agent",
    # model="gemini-2.5-flash",  #gemini-2.5-flash
    # NVIDIA OpenAI 호환 엔드포인트 (openai/ 접두사 + api_base)
    model=LiteLlm(
        model="openai/z-ai/glm-5.3-flash",
        api_base="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("nvidiaapi_key"),
    ),
    description="인사하는 에이전트",
    instruction="""
    당신은 사용자를 환영하고 인사하는 에이전트 입니다.
    사용자의 이름을 물어보고, 그 이름으로 환영 인사를 해주세요.
    """
)



# 개발자 UI 테스트에 접속
# (llm-workspace) PS C:\lab\llm-workspace> cd .\adk-tutorials-main\01-llm-agent\
# (llm-workspace) PS C:\lab\llm-workspace\adk-tutorials-main\01-llm-agent> adk web