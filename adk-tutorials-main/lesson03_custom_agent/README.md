
## 사용자 정의 에이전트 (Custom Agent)
Google ADK에서 기본 제공되는 단방향/반복 워크플로우(`Sequential`, `Loop`, `Parallel`)의 한계를 넘어 **복잡한 비즈니스 로직, 조건부 분기, 외부 API 동기화**를 직접 제어하기 위한 **Custom Agent(사용자 정의 에이전트)** 완벽 가이드입니다.

---

### ① Custom Agent란 무엇인가?

* **정의:** `LlmAgent`나 기본 `Workflow` 클래스를 그대로 사용하는 대신, ADK의 최상위/기본 클래스인 `BaseAgent`를 상속받아 **실행 제어 로직(`_run_async_impl`)을 직접 코딩하여 구현하는 에이전트**입니다.
* **핵심 개념:** LLM의 판단에만 의존하지 않고, 파이썬 코드 레벨에서 상태(State) 값에 따라 에이전트의 실행 경로를 자유롭게 제어할 수 있습니다.

---

### ② 왜 필요한가? 언제 사용해야 하나?

#### 필요성

기본 `LoopAgent`나 `SequentialAgent`는 흐름이 정적으로 고정되어 있거나 단일 조건 검사만 가능합니다. 하지만 실제 서비스에서는 다음과 같이 complex 제어가 필요합니다.

#### 주요 사용 시점

1. **복잡한 다중 조건 분기 (If-Else Branching):**
* 예: 비평 점수가 80점 미만이면 `Writer`로 재진입, 80점 이상 90점 미만이면 `ToneChecker`로 이동, 90점 이상이면 즉시 종료.


2. **동적 파이프라인 변경:**
* 이전 단계의 결과 종류(예: 코드 에러 타입, 카테고리 분류 결과)에 따라 다음에 실행할 하위 에이전트를 동적으로 선택할 때.


3. **외부 시스템/DB 직접 연동:**
* LLM을 거치지 않고 파이썬 코드로 외부 API, DB 저장, 보안 검송 로직을 워크플로우 중간에 직접 삽입할 때.



---

### ③ 사용자 정의(Custom) 에이전트 구현 방법

Custom Agent는 기본적으로 `google.adk.agents.BaseAgent`를 상속하여 작성합니다.

```python
from google.adk.agents import BaseAgent
from google.adk.types import AgentContext, AgentResponse

class MyCustomAgent(BaseAgent):
    def __init__(self, name: str, description: str, sub_agents: list = None):
        super().__init__(name=name, description=description)
        # 하위 에이전트들을 관리 리스트로 등록
        self.sub_agents = sub_agents or []

    async def _run_async_impl(self, context: AgentContext) -> AgentResponse:
        # 여기에 직접 제어할 파이썬 비즈니스 로직을 구현합니다.
        pass

```

---

### ④ `_run_async_impl()` 메서드 구조와 컨텍스트(state) 관리 방식

ADK가 에이전트를 실행할 때 내부적으로 호출하는 핵심 비동기 메서드가 `_run_async_impl()`입니다.

* **`context` (`AgentContext`):**
* 세션 전체의 공유 상태 메모리(`context.state`)에 접근하거나 값을 읽고 쓸 수 있습니다.
* 하위 에이전트를 실행할 때 이 `context`를 그대로 전달하여 상태 데이터를 유지합니다.


* **상태 읽기/쓰기 예시:**
```python
# 1. State에서 이전 결과 가져오기
current_draft = context.state.get("draft_story", "")

# 2. State에 데이터 직접 업데이트하기
context.state["retry_count"] = context.state.get("retry_count", 0) + 1

# 3. 하위 에이전트 실행
sub_response = await self.sub_agents[0].run_async(context)

```



---

### ⑤ 조건 분기, 상태 흐름, 외부 호출 등 사용자 정의 로직 설계

Custom Agent 내부의 일반적인 실행 제어 패턴입니다.

$$\text{User Request} \longrightarrow \text{CustomAgent} \begin{cases} \xrightarrow{\text{1. 초안 작성}} \text{WriterAgent} \\ \xrightarrow{\text{2. 품질 검수}} \text{CriticAgent} \\ \xrightarrow{\text{3. 조건 검사}} \begin{cases} \text{점수 < 80} & \rightarrow \text{Writer로 재이동 (루프)} \\ \text{점수 >= 80} & \rightarrow \text{ToneChecker 실행 후 종료} \end{cases} \end{cases}$$

---

### ⑥ [예제 실습] StoryFlowAgent – 비평/수정/톤 체크 기반 조건부 재생성 흐름 구현

스토리 초안을 작성한 후, 비평 점수를 확인하여 **80점 미만이면 피드백을 반영해 재작성하고, 80점 이상이면 final 톤 검수 에이전트를 거쳐 마무리하는 Custom Agent** 예제입니다.

#### `agent.py`

```python
from google.adk.agents import BaseAgent, LlmAgent
from google.adk.types import AgentContext, AgentResponse

# 1. 하위 에이전트들 정의

# A. 스토리 작성 에이전트
story_writer = LlmAgent(
    name="story_writer",
    model="gemini-2.0-flash",
    instruction="""
    당신은 동화 작가입니다. 사용자의 요청이나 피드백을 바탕으로 매력적인 짧은 이야기를 작성하세요.
    
    [이전 피드백]:
    {critic_feedback}
    """,
    output_key="draft_story"
)

# B. 스토리 비평가 에이전트 (점수와 피드백 부여)
story_critic = LlmAgent(
    name="story_critic",
    model="gemini-2.0-flash",
    instruction="""
    다음 동화를 검토하고, 평가 점수(0~100)와 개선 피드백을 작성하세요.

    [검토할 동화]:
    {draft_story}

    [출력 규칙]:
    첫 번째 줄에는 반드시 'SCORE: [점수]' 형식으로 적고, 둘째 줄부터 피드백을 적으세요.
    예시:
    SCORE: 75
    전개는 좋으나 결말이 조금 급작스럽습니다.
    """,
    output_key="critic_feedback"
)

# C. final 톤 교정 에이전트 (80점 이상일 때만 호출)
tone_polisher = LlmAgent(
    name="tone_polisher",
    model="gemini-2.0-flash",
    instruction="""
    작성된 동화의 어조를 더 따뜻하고 감성적인 다듬어진 문장으로 최종 리팩토링하세요.

    [원본 동화]:
    {draft_story}
    """,
    output_key="final_story"
)

# 2. Custom Agent 클래스 정의
class StoryFlowAgent(BaseAgent):
    def __init__(self, name: str, description: str):
        super().__init__(name=name, description=description)
        # 하위 에이전트 매핑
        self.writer = story_writer
        self.critic = story_critic
        self.polisher = tone_polisher

    async def _run_async_impl(self, context: AgentContext) -> AgentResponse:
        max_retries = 3
        # State 초기화
        context.state["critic_feedback"] = "최초 작성"

        for i in range(max_retries):
            # Step 1: 초안 작성
            await self.writer.run_async(context)

            # Step 2: 비평 및 점수 산정
            await self.critic.run_async(context)
            feedback = context.state.get("critic_feedback", "")

            # 점수 파싱 (예: SCORE: 85 -> 85)
            score = 0
            for line in feedback.splitlines():
                if "SCORE:" in line:
                    try:
                        score = int(line.split("SCORE:")[1].strip())
                    except ValueError:
                        score = 50
                    break

            # Step 3: 조건 분기 제어
            if score >= 80:
                # 80점 이상인 경우 톤 교정 후 완료
                print(f"[CustomAgent] 점수 통과 ({score}점). 톤 교정 단계를 진행합니다.")
                return await self.polisher.run_async(context)
            else:
                print(f"[CustomAgent] 점수 미달 ({score}점). 재작성을 진행합니다. ({i+1}/{max_retries})")

        # 최대 재시도 후에도 미달 시 마지막 초안 상태로 톤 교정 수행
        return await self.polisher.run_async(context)


# 3. 루트 에이전트 지정
root_agent = StoryFlowAgent(
    name="story_flow_agent",
    description="조건부 점수 평가 및 톤 교정이 포함된 커스텀 스토리 생성 에이전트"
)

```

---

### ⑦ 실행 및 검증 방법

1. 위 작성된 `agent.py` 코드를 저장하고 실행합니다.
```bash
adk web

```


2. UI에서 `story_flow_agent`를 선택한 후 다음 요청을 보냅니다.
> `"용감한 아기 곰이 숲속 보물을 찾는 짧은 동화를 써줘."`


3. 타임라인 Traces에서 점수 미달 시 `story_writer` $\rightarrow$ `story_critic` 루프가 실행되다가, 80점 이상을 달성하는 순간 `tone_polisher`로 이동하여 최종 동화가 생성되는 흐름을 확인할 수 있습니다.