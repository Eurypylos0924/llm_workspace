
Google Cloud Next 2025에서 공개된 Google ADK(Agent Development Kit, 에이전트 개발 키트)는 기업 및 개발자가 **독자적이고 안정적인 AI 에이전트를 구축, 디버깅, 배포 및 오케스트레이션**할 수 있도록 지원하는 **오픈소스 AI 에이전트 개발 프레임워크**입니다.

단순한 대화형 AI(챗봇)를 넘어, 복잡한 업무 프로세스를 스스로 판단하고 외부 시스템과 연동하여 수행하는 **자율형 멀티 에이전트 시스템** 구축에 최적화되어 있습니다.

---

### ① ADK에서 제공하는 에이전트 유형 3가지

* **LLM Agent:** 대화 모델(Gemini 등)을 중심 엔진으로 삼아 자연어를 이해하고, 판단하고, 필요한 도구를 직접 골라 실행하는 자율형 에이전트입니다.
* **Workflow Agent:** 실행 순서가 명확히 지정된 에이전트입니다. 순차적(`Sequential`), 병렬적(`Parallel`), 조건 반복(`Loop`) 구조로 하위 에이전트의 흐름을 안전하게 제어합니다.
* **Custom Agent:** 표준 프레임워크 범주를 벗어나 고유한 알고리즘이나 외부 시스템과의 특수 인터페이스를 연결할 때 직접 클래스를 상속받아 구현하는 에이전트입니다.

---

### ② LLM(Large Language Model)의 개념과 역할

* **개념:** 수많은 텍스트 데이터를 학습하여 인간의 언어를 이해하고 생성할 수 있는 대규모 인공지능 신경망 모델입니다 (예: Gemini 1.5 Pro/Flash).
* **역할:** 에이전트 시스템에서 **'두뇌'** 역할을 담당합니다. 사용자 의도 파악, 추론(Reasoning), 답변 생성, 그리고 어떤 도구를 실행할지 결정하는 판단 기준이 됩니다.

---

### ③ LLM 에이전트란 무엇이며, 어떻게 작동하는가?

* **개념:** 단순히 질문에 답만 하는 LLM에 도구(Tools), 기억(Memory), 목표(Goal)를 부여하여 스스로 작업을 완수하도록 확장한 시스템입니다.
* **작동 루프:**
1. **입력 받기:** 사용자의 목표 메시지 수신
2. **추론(Thought):** 목표 달성을 위해 다음 행동 계획 수립
3. **도구 실행(Action):** 필요 시 외부 API, DB 조회, 계산기 등 실행
4. **결과 확인(Observation):** 실행 결과를 해석하여 최종 답변 혹은 추가 행동 결정



---

### ④ Google ADK로 LLM 에이전트 구현하기 (실습 준비)

#### 1. 필수 사전 준비

* Python 3.10 이상 설치
* Google AI Studio에서 **Gemini API Key** 발급

#### 2. 패키지 설치

```bash

uv add "openai>=2.20,<3"
uv add "google-adk[extensions]"

```

---

### ⑤ 에이전트 폴더 구조 및 설정 규칙

ADK는 규칙 기반 모듈 로딩 방식을 권장합니다. 에이전트 프로젝트는 기본적으로 아래와 같은 구조를 갖춥니다.

```text
my_adk_app/
├── .env                  # API 키 등 환경변수 설정
├── main.py               # 실행 엔트리포인트
└── agents/               # 에이전트 모듈 폴더
    └── greeting_agent/   # '인사하는 에이전트' 모듈
        ├── __init__.py   # 모듈 내보내기 정의
        └── agent.py      # 에이전트 핵심 로직 및 프롬프트

```

**.env 파일 예시:**

```env
GEMINI_API_KEY=your_gemini_api_key_here

```

---

### ⑥ 인사하는 LLM 에이전트 직접 만들어보기 (코드)

#### `agents/greeting_agent/agent.py`

```python
from google.adk.agents import LlmAgent

# 친절하게 인사하고 자기소개를 하는 LLM 에이전트 정의
greeting_agent = LlmAgent(
    name="GreetingAgent",
    model="gemini-1.5-flash",
    instruction="""
    당신은 사용자에게 친절하게 인사하는 AI 비서입니다.
    사용자가 말을 걸면 정중하고 따뜻하게 인사하고, 어떤 도움이 필요한지 물어보세요.
    답변은 한국어로 2-3문장 이내로 명확하게 작성하세요.
    """
)

```

#### `agents/greeting_agent/__init__.py`

```python
from .agent import greeting_agent

__all__ = ["greeting_agent"]

```

#### `main.py`

```python
import os
from dotenv import load_dotenv
from agents.greeting_agent import greeting_agent

load_dotenv()

if __name__ == "__main__":
    response = greeting_agent.run("안녕하세요! 오늘 날씨가 참 좋네요.")
    print("Agent Response:\n", response.text)

```

---

### ⑦ ADK Web UI를 활용한 테스트 방법

ADK는 터미널 모드 외에도 시각적으로 대화 내역과 에이전트 상태를 검증할 수 있는 개발용 Web UI(Debug Console)를 제공합니다.

1. **Web UI 서버 실행:**
```bash
adk web --agents-dir ./agents

```


2. **브라우저 접속:** `http://localhost:8000` 로 이동합니다.
3. **테스트 진행:**
* 드롭다운 메뉴에서 `GreetingAgent`를 선택합니다.
* 채팅 창에 메시지를 입력하여 응답 속도, 프롬프트 적용 결과, 추론 상태를 실시간으로 확인합니다.

uv run adk web 로 작동