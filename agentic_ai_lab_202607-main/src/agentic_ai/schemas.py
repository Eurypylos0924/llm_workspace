"""Structured Output 검증에 사용하는 Pydantic 모델."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# LLM의 응답을 검증하기 위한 스키마로, 구조화된 출력의 기준이 된다.
class InputAnalysis(BaseModel):
    """사용자 입력을 분석한 결과를 담는 Schema.

    LLM이 생성한 JSON 문자열은 이 Schema를 기준으로 검증된 뒤에만
    애플리케이션에서 신뢰할 수 있는 데이터로 취급한다.
    """

    model_config = ConfigDict(extra="forbid", strict=True)

    input_type: Literal["question", "document", "calculation", "unknown"]
    summary: str
    keywords: list[str]
    confidence: float = Field(ge=0.0, le=1.0)
