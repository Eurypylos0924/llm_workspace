from google.adk import Context, Event, Workflow
from google.adk.workflow import node

from .sub_agents.writer import initial_writer_agent
from .sub_agents.critic import critic_agent
from .sub_agents.refiner import refiner_agent


@node(rerun_on_resume=True)
async def refinement_orchestrator(
    ctx: Context,
    node_input: str,
):
    """
    Critic → Refiner 과정을 최대 5회 반복하는
    동적 Workflow Orchestrator이다.
    """

    current_text = node_input

    # 최대 5회 반복
    for iteration in range(5):

        # --------------------------------------------------
        # 1. Critic Agent 실행
        # --------------------------------------------------
        critique = await ctx.run_node(
            critic_agent,
            node_input=current_text,
        )

        # --------------------------------------------------
        # 2. Refiner Agent 실행
        # --------------------------------------------------
        current_text = await ctx.run_node(
            refiner_agent,
            node_input=current_text,
        )

        # --------------------------------------------------
        # 진행 상황을 State에 저장
        # --------------------------------------------------
        yield Event(
            state={
                "iteration": iteration + 1,
                "critique": critique,
                "current_text": current_text,
            }
        )

    # ------------------------------------------------------
    # 최종 결과
    # ------------------------------------------------------
    yield Event(
        output=current_text
    )


root_agent = Workflow(
    name="반복적글쓰기파이프라인",

    description=(
        "초기 글을 작성한 후 "
        "Critic과 Refiner를 최대 5회 반복하여 "
        "글의 완성도를 높이는 Workflow이다."
    ),

    edges=[
        (
            "START",
            initial_writer_agent,
            refinement_orchestrator,
        ),
    ],
)


# (llm-workspace) PS C:\lab\llm-workspace\adk-tutorials-main\02-workflow-agent> cd 2-loop-agent
# (llm-workspace) PS C:\lab\llm-workspace\adk-tutorials-main\02-workflow-agent\2-loop-agent> adk web

#  글쓰기를 해줘.