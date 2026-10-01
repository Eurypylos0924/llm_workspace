워크플로우 에이전트(Workflow Agent)의 핵심 개념부터 3단계 코드 개선 파이프라인(작성 → 리뷰 → 리팩토링)을 순차적으로 다루는 가이드입니다.

---

### 1 워크플로우 에이전트란 무엇인가?

* **정의:** 여러 개의 에이전트를 특정 순서나 로직에 따라 연결하여, 복잡하고 연쇄적인 업무 프로세스를 제어 및 오케스트레이션(Orchestration)하는 에이전트입니다.
* **필요성:** 단일 LLM 에이전트에게 너무 복잡하거나 긴 작업을 맡기면 환각(Hallucination)이 발생하거나 지시 사항을 누락하기 쉽습니다. 이를 전문화된 단계를 나누어 처리하도록 관리합니다.

---

### 2 일반 에이전트 vs 워크플로우 에이전트 차이점

| 구분 | 일반 LLM 에이전트 (LlmAgent) | 워크플로우 에이전트 (SequentialAgent 등) |
| --- | --- | --- |
| **특성** | **비결정론적 (Non-Deterministic)** | **결정론적 (Deterministic) 제어** |
| **행동 방식** | LLM이 프롬프트를 보고 다음 행동이나 도구(Tool) 호출 여부를 스스로 판단 | 정해진 순서/조건/반복 제어 흐름에 따라 하위 에이전트를 강제로 실행 |
| **제어권** | LLM 모델의 추론 엔진 | 개발자가 정의한 워크플로우 파이프라인 |
| **주요 용어** | 대화, 자율적 도구 선택 | 단계별 순차 실행, 병렬 실행, 루프 |

---


### 3 ADK 워크플로우 에이전트 (Workflow Agent) 3가지 유형

워크플로우 에이전트는 여러 하위 에이전트의 실행 순서와 흐름을 통제하기 위해 사용되며, 실행 방식에 따라 다음과 같이 3가지 유형으로 나뉩니다.

#### ① Sequential Agent (순차 에이전트)

* **실행 방식:** **순차 실행**

* **특징:** 하위 에이전트들을 파이프라인 형태(① $\rightarrow$ ② $\rightarrow$ ③)로 나열하여 순서대로 실행합니다. 앞선 에이전트의 결과물(Output)이 다음 에이전트의 입력(Input)으로 자동 전달됩니다.


* **활용 예시:** 코드 작성 $\rightarrow$ 코드 리뷰 $\rightarrow$ 리팩토링 과정처럼 단계별 진행이 필수적인 작업

#### ② Loop Agent (루프 에이전트)

* **실행 방식:** **반복 실행**

* **특징:** 특정 만족 조건(또는 최대 반복 횟수)을 충족할 때까지 하위 에이전트의 순환 작업을 반복합니다.


* **활용 예시:** 코드 검증 중 에러가 해결될 때까지 '수정 $\rightarrow$ 테스트' 과정을 반복하는 작업

#### ③ ParallelAgent (병렬 에이전트)

* **실행 방식:** **병렬(동시) 실행**

* **특징:** 여러 개의 하위 에이전트를 동시에 실행하여 각각의 결과를 취합합니다. 작업 시간이 짧아지고 효율성이 극대화됩니다.


* **활용 예시:** 동일한 주제에 대해 '뉴스 검색', 'SNS 반응 수집', '학술 자료 조사'를 동시에 진행하는 작업

### 4 SequentialAgent 개념 및 구조 이해

`SequentialAgent`는 하위 에이전트들을 파이프라인 형태로 나열하여 **앞선 에이전트의 출력(Output)을 다음 에이전트의 입력(Input)으로 자동 전달**하는 워크플로우 에이전트입니다.

* **입출력 데이터 흐름:**

$$\text{User Input} \longrightarrow \text{[Agent 1]} \xrightarrow{\text{Output 1}} \text{[Agent 2]} \xrightarrow{\text{Output 2}} \text{[Agent 3]} \longrightarrow \text{Final Output}$$



---

### 5 3단계 에이전트 파이프라인 실습 (작성 → 리뷰 → 리팩토링)

`SequentialAgent`를 활용하여 파이썬 코드를 작성하고, 리뷰하고, 최종 개선하는 3단계 파이프라인을 구축해 보겠습니다.

#### `pipeline_agent.py`

```python
from google.adk.agents import LlmAgent, SequentialAgent

# 1단계: 파이썬 코드 작성 에이전트
coder_agent = LlmAgent(
    name="coder_agent",
    model="gemini-2.0-flash",
    description="요청사항에 맞는 파이썬 코드를 작성합니다.",
    instruction="""
    당신은 숙련된 파이썬 개발자입니다.
    사용자의 요구사항을 받아 완전하고 동작 가능한 파이썬 코드를 작성하세요.
    코드 외의 부연 설명은 최소화하고 코드 블록 중심으로 작성하세요.
    """
)

# 2단계: 코드 리뷰 에이전트
reviewer_agent = LlmAgent(
    name="reviewer_agent",
    model="gpt-4o-mini",
    description="작성된 코드를 리뷰하여 문제점과 개선점을 찾습니다.",
    instruction="""
    당신은 깐깐한 시니어 코드 리뷰어입니다.
    전달받은 파이썬 코드를 분석하여 다음 항목을 검토하세요:
    1. 버그 및 예외 처리 누락 여부
    2. 가독성 및 PEP 8 스타일 가이드 준수 여부
    3. 성능 개선 포인트
    개선해야 할 점을 구체적인 항목별로 정리해주세요.
    """
)

# 3단계: 코드 리팩토링 에이전트
refactor_agent = LlmAgent(
    name="refactor_agent",
    model="gemini-2.0-flash",
    description="리뷰 결과를 바탕으로 최적화된 최종 코드를 작성합니다.",
    instruction="""
    당신은 코드 리팩토링 전문가입니다.
    이전 단계의 '원본 코드'와 '코드 리뷰 내용'을 바탕으로,
    리뷰 피드백이 완벽히 반영된 최상의 최종 파이썬 코드를 제시하세요.
    최종 코드 아래에는 주요 개선 사항을 요약해 주세요.
    """
)

# 3개의 에이전트를 순차적으로 실행하는 SequentialAgent 정의
code_pipeline_agent = SequentialAgent(
    name="code_pipeline_agent",
    description="코드 작성 -> 리뷰 -> 리팩토링을 순차 처리하는 파이프라인",
    sub_agents=[coder_agent, reviewer_agent, refactor_agent]
)

```

---

### 6 ADK Web UI로 에이전트 테스트

구축한 파이프라인을 시각적으로 확인하고 실행 단계별 입출력을 검증합니다.

(1) **프로젝트 폴더 구조 확인:**
```text
my_project/
├── .env
└── agents/
    └── code_pipeline/
        ├── __init__.py
        └── agent.py (code_pipeline_agent 정의)

```


(2) **Web UI 서버 실행:**
터미널에서 아래 명령어를 입력합니다.
```bash
adk webui --agents-dir ./agents

```


(3) **테스트 진행:**
* 브라우저(`http://localhost:8000`)에 접속합니다.
* 에이전트 목록에서 `code_pipeline_agent`를 선택합니다.
* 입력창에 테스트 요청을 전달합니다:
> `"문자열 리스트를 입력받아 중복을 제거하고 알파벳 순으로 정렬하는 함수를 만들어줘."`


* Web UI 우측 타임라인에서 `coder_agent` $\rightarrow$ `reviewer_agent` $\rightarrow$ `refactor_agent`로 단계별 출력이 이어지는 과정을 실시간으로 확인할 수 있습니다.


## 6. Google ADK의 LoopAgent(루프 에이전트)

---

### ① LoopAgent란 무엇인가?

* **정의:** 특정 종료 조건(Completion Condition)을 만족하거나 최대 반복 횟수(Max Iterations)에 도달할 때까지 하위 에이전트들을 순환하여 반복 실행하는 워크플로우 에이전트입니다.
* **필요성:** 한 번의 실행(Single Pass)만으로 완성도 높은 결과물을 얻기 어려운 **품질 검수, 피드백 반영, 자기 수정(Self-Correction)** 작업에 필수적인 구조입니다.

---

### ② 반복 제어 방식: 반복 횟수 지정 vs 종료 조건 감지

LoopAgent의 루프 탈출 제어 방식은 크게 두 가지로 나뉩니다.

1. **최대 반복 횟수 지정 (`max_iterations`):**
* 루프가 무한히 실행되는 것(Infinite Loop)을 방지하는 안전장치입니다.
* 지정된 횟수(예: 3회, 5회)만큼 하위 에이전트 순환이 끝나면 조건 만족 여부와 관계없이 작업을 종료합니다.


2. **종료 조건 감지 (Condition Detection):**
* 하위 에이전트(비평가 에이전트)가 평가 후 특정 신호(예: `PASSED`, `APPROVED` 키워드 또는 `is_finished=True` 상태 값)를 반환했는지 검사하여 동적으로 루프를 탈출합니다.



---

### ③ ADK LoopAgent의 동작 구조 및 특징

* **데이터 연동 (State Share):**
* 루프 내의 모든 하위 에이전트들은 ADK 세션 상태(`State`)를 공유합니다.
* `작성자 Agent`가 저장한 `output_key` 결과물을 `비평가 Agent`가 받아 검토하고, 다시 `작성자 Agent`가 피드백을 반영하는 구조입니다.


* **실행 순서:**

$$\text{[Loop 시작]} \longrightarrow \text{1. 초안 작성} \longrightarrow \text{2. 품질 평가} \longrightarrow \text{3. 종료 조건 검사} \begin{cases} \text{만족 시} & \rightarrow \text{루프 종료} \\ \text{불만족 시} & \rightarrow \text{1단계로 재진입 (피드백 반영)} \end{cases}$$



---


### ④ 테스트 방법

1. 위 작성된 `agent.py` 파일 및 `.env` 파일 설정 후 ADK Web UI를 실행합니다.
```bash
adk web

```


2. 웹 UI에서 `article_loop_agent`를 선택하고 아래 요청을 보냅니다.
> `"Google Cloud Next 2025에서 발표된 ADK에 대한 소개글을 작성해줘."`


3. 타임라인 이벤트 창에서 `writer_agent` $\rightarrow$ `critic_agent` 흐름이 수행되고, `PASSED` 신호가 나올 때까지 반복(최대 3회) 후 최종 승인된 글이 출력되는 과정을 확인할 수 있습니다.


Google ADK의 ParallelAgent(병렬 에이전트)에 대한 개념부터 멀티 모듈 수집 파이프라인 실습 가이드까지 정리해 드립니다.

---
## 7. Google ADK의 ParallelAgent(병렬 에이전트)
### ① ParallelAgent란?

* **정의:** 여러 개의 하위 에이전트(Sub-Agents)를 **동시에(병렬로) 실행**하고, 각 에이전트의 작업 결과를 하나로 수집 및 통합하는 워크플로우 에이전트입니다.
* **필요성:** 순차적 처리가 필요 없는 독립적인 데이터 수집, 다각도 분석, 다국어 번역 등의 작업에서 처리 시간을 극적으로 단축할 수 있습니다.

---

### ② 동작 방식 및 병렬 실행 구조 이해

ParallelAgent는 하위 에이전트들이 서로의 출력에 의존하지 않는 **독립적인 작업 단위**일 때 사용합니다.

* **실행 구조:**

$$\text{User Request} \longrightarrow \text{[ParallelAgent]} \begin{cases} \xrightarrow{\text{동시 실행}} \text{[Agent A (IT 뉴스 수집)]} \xrightarrow{\text{Result A}} \\ \xrightarrow{\text{동시 실행}} \text{[Agent B (시장 분석)]} \xrightarrow{\text{Result B}} \\ \xrightarrow{\text{동시 실행}} \text{[Agent C (SNS 반응)]} \xrightarrow{\text{Result C}} \end{cases} \longrightarrow \text{Merged Results}$$


* **Sequential vs Parallel 차이:**
* **SequentialAgent:** Agent A의 출력이 Agent B의 입력으로 사용됨 (순차 의존적)
* **ParallelAgent:** Agent A, B, C가 서로 관여하지 않고 동시에 구동됨 (시간 절약)



---

### ③ 독립 실행 하위 에이전트 구성 및 관리 방식

1. **상태(State) 격리 및 키 지정:**
* 각 하위 에이전트가 동시에 결과를 기록할 때 데이터가 덮어씌워지지 않도록, 서로 다른 `output_key`를 지정해야 합니다.


2. **에이전트 역할 분리:**
* 하위 에이전트는 특화된 단일 역할만 부여하여 프롬프트 명확성을 확보합니다.



---

### ④ [예제 실습] 웹 리서치 및 데이터 수집용 병렬 에이전트 구현

기술 동향 리서치를 위해 **IT 뉴스 수집**, **시장 영향 분석**, **기술 스택 분석**을 동시에 수행하는 병렬 파이프라인 예시 코드입니다.

#### `agent.py`

```python
from google.adk.agents import LlmAgent, ParallelAgent

# 1. IT 기술 뉴스 요약 에이전트
news_agent = LlmAgent(
    name="news_agent",
    model="gemini-2.0-flash",
    description="최신 IT 뉴스 및 최신 동향을 요약합니다.",
    instruction="""
    주제에 관련된 최신 기술 트렌드와 대표적인 소식 3가지를 정리하세요.
    """,
    output_key="news_summary"  # 고유 키 지정
)

# 2. 시장 및 비즈니스 영향 분석 에이전트
market_agent = LlmAgent(
    name="market_agent",
    model="gemini-2.0-flash",
    description="기술이 비즈니스 시장에 미치는 영향을 분석합니다.",
    instruction="""
    주제 기술이 관련 산업 및 기업 비즈니스에 미치는 긍정적 영향과 위험 요소를 요약하세요.
    """,
    output_key="market_analysis"  # 고유 키 지정
)

# 3. 개발 기술 스택 분석 에이전트
tech_agent = LlmAgent(
    name="tech_agent",
    model="gemini-2.0-flash",
    description="관련 주요 기술 스택과 프레임워크를 정리합니다.",
    instruction="""
    주제 기술을 도입할 때 핵심이 되는 주요 기술 스택, 프레임워크, 라이브러리를 정리하세요.
    """,
    output_key="tech_stack"  # 고유 키 지정
)

# 4. 3개 에이전트를 동시 실행하는 ParallelAgent 정의
parallel_research_agent = ParallelAgent(
    name="parallel_research_agent",
    description="뉴스, 시장, 기술 분석을 동시에 병렬로 수집하는 리서치 에이전트",
    sub_agents=[news_agent, market_agent, tech_agent]
)

root_agent = parallel_research_agent

```

---

### ⑤ ADK Web UI 상에서 실행 테스트 및 결과 확인

1. **프로젝트 실행:**
터미널에서 ADK Web UI를 실행합니다.
```bash
adk web

```


2. **테스트 진행:**
* 웹 브라우저(`http://localhost:8000`)에 접속하여 `parallel_research_agent`를 선택합니다.
* 입력창에 리서치 주제를 입력합니다:
> `"생성형 AI 에이전트(Agentic AI) 기술 동향"`




3. **결과 확인 포인트:**
* **동시 실행 로그:** 우측 타임라인에서 `news_agent`, `market_agent`, `tech_agent`가 직렬 순서가 아닌 **동일한 타임스탬프에 동시 호출**되는 것을 확인할 수 있습니다.
* **State 수집 확인:** 대화 세션 상태 메모리에 `news_summary`, `market_analysis`, `tech_stack` 3가지 키로 데이터가 각각 안전하게 저장 및 통합 출력됩니다.