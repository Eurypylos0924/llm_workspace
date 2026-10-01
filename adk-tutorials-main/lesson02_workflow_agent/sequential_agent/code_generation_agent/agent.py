# from google.adk.agents import SequentialAgent  # 이전 버전 방법
from google.adk import Workflow
from google.adk.workflow import START

from .sub_agents.writer import code_writer_agent
from .sub_agents.reviewer import code_reviewer_agent
from .sub_agents.refactor import code_refactor_agent


# 순차적에이전트가 고정적이기 때문에 결정론적 에이전트로 model과 instruction은 여기에서는 필요없다.
code_pipeline_agent = Workflow(
    name= "code_pipeline_agent",
    description= "코드작성 > 리뷰 > 리팩토링의 순서로 작업을 수행하는 시퀀셜 에이전트입니다",
   edges=[
        (START, code_writer_agent),
        (code_writer_agent, code_reviewer_agent),
        (code_reviewer_agent, code_refactor_agent),
    ],
)

root_agent = code_pipeline_agent

# (llm-workspace) PS C:\lab\llm-workspace\adk-tutorials-main\02-workflow-agent> cd .\1-sequntial-agent\
# (llm-workspace) PS C:\lab\llm-workspace\adk-tutorials-main\02-workflow-agent\1-sequntial-agent> adk web

# 계산기 코드를 작성해줘.