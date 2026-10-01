from google.adk import Workflow
from google.adk.workflow import START, JoinNode

from .sub_agents.renewable_energy import renewable_energy_agent
from .sub_agents.ev_technology import ev_technology_agent
from .sub_agents.carbon_capture import carbon_capture_agent
from .sub_agents.synthesizer import synthesis_agent


research_join = JoinNode(
    name="research_results_join"
)


root_agent = Workflow(
    name="sustainable_technology_research_pipeline",

    description=(
        "재생 에너지, 전기차 기술, 탄소 포집 기술을 "
        "병렬로 조사한 후 결과를 통합하여 "
        "지속가능 기술 동향 보고서를 생성하는 Workflow이다."
    ),

    edges=[
        (
            START,

            (
                renewable_energy_agent,
                ev_technology_agent,
                carbon_capture_agent,
            ),

            research_join,

            synthesis_agent,
        )
    ],
)


# (llm-workspace) PS C:\lab\llm-workspace\adk-tutorials-main\02-workflow-agent> cd .\3-parallel-agent\
# (llm-workspace) PS C:\lab\llm-workspace\adk-tutorials-main\02-workflow-agent\3-parallel-agent> adk web

# 연구를 시작해 주세요.