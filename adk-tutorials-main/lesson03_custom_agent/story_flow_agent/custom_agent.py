from google.adk.agents import BaseAgent, LlmAgent
from google.adk.agents.invocation_context import InvocationContext
from google.adk.events import Event

from typing import AsyncGenerator
from typing_extensions import override


class StoryFlowAgent(BaseAgent):
    """
    스토리 생성 및 수정 워크플로우를 처리하는 사용자 정의 에이전트이다.

    실행 흐름

    1. 스토리 초안 생성
    2. 비평
    3. 수정
    4. 비평/수정 반복
    5. 문법 검사
    6. 톤 검사
    7. 톤이 negative이면 스토리 재생성
    """

    # --------------------------------------------------
    # 내부 에이전트
    # --------------------------------------------------

    story_generator: LlmAgent
    critic: LlmAgent
    reviser: LlmAgent
    grammar_check: LlmAgent
    tone_check: LlmAgent

    # --------------------------------------------------
    # Pydantic 설정
    # --------------------------------------------------

    model_config = {
        "arbitrary_types_allowed": True
    }

    def __init__(
        self,
        name: str,
        story_generator: LlmAgent,
        critic: LlmAgent,
        reviser: LlmAgent,
        grammar_check: LlmAgent,
        tone_check: LlmAgent,
    ):
        """
        StoryFlowAgent를 초기화한다.
        """

        super().__init__(
            name=name,
            description="스토리를 생성하고 비평, 수정, 문법 및 톤 검사를 수행하는 사용자 정의 Workflow 에이전트이다.",
            story_generator=story_generator,
            critic=critic,
            reviser=reviser,
            grammar_check=grammar_check,
            tone_check=tone_check,

            # BaseAgent의 sub_agents에 등록한다.
            sub_agents=[
                story_generator,
                critic,
                reviser,
                grammar_check,
                tone_check,
            ],
        )

    @override
    async def _run_async_impl(
        self,
        ctx: InvocationContext,
    ) -> AsyncGenerator[Event, None]:
        """
        사용자 정의 Workflow 실행 로직이다.
        """

        # ==================================================
        # 1. 초기 스토리 생성
        # ==================================================

        print("\n[1] 스토리 초안 생성")

        async for event in self.story_generator.run_async(ctx):
            yield event

        # --------------------------------------------------
        # 스토리가 생성되었는지 확인
        # --------------------------------------------------

        current_story = ctx.session.state.get("current_story")

        if not current_story:
            print("스토리 생성에 실패하여 Workflow를 종료한다.")
            return

        # ==================================================
        # 2. 비평 → 수정 반복
        # ==================================================

        max_iterations = 2

        for iteration in range(max_iterations):

            print(
                f"\n[2] 비평/수정 반복 "
                f"{iteration + 1}/{max_iterations}"
            )

            # --------------------------------------------------
            # 2-1. 비평
            # --------------------------------------------------

            print("  └─ critic 실행")

            async for event in self.critic.run_async(ctx):
                yield event

            # --------------------------------------------------
            # 2-2. 수정
            # --------------------------------------------------

            print("  └─ reviser 실행")

            async for event in self.reviser.run_async(ctx):
                yield event

        # ==================================================
        # 3. 문법 검사
        # ==================================================

        print("\n[3] 문법 검사")

        async for event in self.grammar_check.run_async(ctx):
            yield event

        # ==================================================
        # 4. 톤 검사
        # ==================================================

        print("\n[4] 톤 검사")

        async for event in self.tone_check.run_async(ctx):
            yield event

        # ==================================================
        # 5. 톤 검사 결과 확인
        # ==================================================

        tone_check_result = ctx.session.state.get(
            "tone_check_result"
        )

        print(
            f"\n[5] 톤 검사 결과: "
            f"{tone_check_result}"
        )

        # ==================================================
        # 6. 부정적인 톤이면 스토리 재생성
        # ==================================================

        if tone_check_result == "negative":

            print(
                "\n[6] 부정적인 톤이므로 "
                "스토리를 재생성한다."
            )

            async for event in self.story_generator.run_async(ctx):
                yield event

        else:

            print(
                "\n[6] 긍정적/중립적 톤이므로 "
                "현재 스토리를 유지한다."
            )