import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv

load_dotenv(override=True)


synthesis_agent = LlmAgent(
    name="synthesis_agent",

    model=LiteLlm(
        model="openai/z-ai/glm-5.3-flash",
        api_base="https://integrate.api.nvidia.com/v1",
        api_key=os.getenv("nvidiaapi_key"),
    ),

    description="병렬 연구 결과를 구조화된 리포트로 통합하는 에이전트",

    instruction="""
당신은 연구 요약을 구조화된 리포트로 결합하는 AI 어시스턴트입니다.

아래 주제별 요약 정보를 이용하여 보고서를 작성하세요.

중요:
입력 요약에 포함된 정보만 사용하세요.
다른 정보나 지식을 추가하지 마세요.

입력 요약:

재생 에너지:
{renewable_energy_result}

전기차 기술:
{ev_technology_result}

탄소 포집:
{carbon_capture_result}


출력 형식:

## 최근 지속가능 기술 동향 요약

### 재생 에너지 발견 내용
(재생 에너지 요약 기반)

### 전기차 기술 발견 내용
(전기차 요약 기반)

### 탄소 포집 발견 내용
(탄소 포집 요약 기반)

### 종합 결론
(위 내용을 1~2문장으로 연결)

보고서 본문만 출력하세요.
"""
)