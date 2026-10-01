import os
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from dotenv import load_dotenv
from .custom_agent import StoryFlowAgent

load_dotenv(override=True)

# Gemini 대신 NVIDIA 모델 사용 (5개 에이전트가 같은 모델을 공유)
NVIDIA_MODEL = LiteLlm(
    model="openai/z-ai/glm-5.3-flash",
    api_base="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("nvidiaapi_key"),
)

# --- 개별 LLM 에이전트 정의 ---

# 스토리 생성 에이전트
TOPIC = "미래의 세계" # 스토리 생성 주제 (원하는 스토리가 있으면 여기를 변경해주시면 됩니다)
story_generator = LlmAgent(
    name = "story_generator",
    model = NVIDIA_MODEL, # 사용할 모델 (필요에 따라 변경)
    instruction = f"""
    당신은 스토리 작가입니다.
    사용자가 제공한 '주제 : {TOPIC}'를 기반으로 100단어 정도의 짧은 스토리를 작성하세요.
    """,
    output_key = "current_story" # 세션 상태에 저장되는 출력 키
)

# 스토리 비평 에이전트
critic = LlmAgent(
    name = "critic",
    model = NVIDIA_MODEL,
    instruction = """
    당신은 스토리 비평가 입니다. 세션 상태에서 'current_story'로 제공된 스토리를 리뷰하고,
    스토리를 개선할 수 있는 1~2문자의 건설적인 비평을 제공하세요.
    줄거리나 캐릭터에 집중하세요.
    """,
    output_key = "criticism" # 비평을 저장할 세션 상태의 출력 키
)

# 스토리 수정 에이전트
reviser = LlmAgent(
    name = "reviser",
    model = NVIDIA_MODEL,
    instruction = """
    당신은 스토리 수정자 입니다. 세션 상태에서 'criticism'에 기반해 'current_story'를 수정하세요.
    수정된 스토리만 출력하세요.
    """, 
    output_key = "current_story" # 원래의 스토리를 덮어쓸 키
)

# 문법 검사 에이전트
grammar_check = LlmAgent(
    name = "grammar_check",
    model = NVIDIA_MODEL,
    instruction = """
    당신은 문법 검사기 입니다. 세션 상태에서 'current_story'로 제공된 스토리의 문법을 검사하고, 수정 사항을 제시하세요.
    오류가 없으면 'Grammar is good!'을 출력하세요
    """,
    output_key = "grammar_suggestions"
)

# 톤 분석 에이전트
tone_check = LlmAgent(
    name = "tone_check",
    model = NVIDIA_MODEL,
    instruction = """
    당신은 톤 분석가 입니다. 세션 상태에서 'current_story'로 제공된 스토리의 톤을 분석하고, 다음 중 하나의 단어만 출력하세요:
    'positive' (긍적적), 'negative' (부정적)
    """,
    output_key = "tone_check_result" # 이 에이전트의 출력은 조건부 흐름을 결정합니다.
)

# --- 사용자 정의 에이전트 인스턴스 생성 ---

root_agent = StoryFlowAgent(
    name = "story_flow_agent",
    story_generator = story_generator,
    critic = critic,
    reviser = reviser,
    grammar_check = grammar_check,
    tone_check = tone_check
)



# (llm-workspace) PS C:\lab\llm-workspace\adk-tutorials-main> cd .\03-custom-agent\
# (llm-workspace) PS C:\lab\llm-workspace\adk-tutorials-main\03-custom-agent> adk web

# 시작해 주세요.