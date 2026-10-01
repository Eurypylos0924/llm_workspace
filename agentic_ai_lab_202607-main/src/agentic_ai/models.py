"""Chat Model과 Embedding Model Factory."""

from __future__ import annotations

from .config import get_settings


# OpenAI API 키가 필요한 모델 생성 전에 필수 설정을 먼저 확인한다.
def _require_api_key() -> str:
    settings = get_settings()
    if not settings.openai_api_key:
        raise RuntimeError(
            "OPENAI_API_KEY가 설정되지 않았습니다. "
            "agentic-ai-notebooks/.env 파일을 만들고 API Key를 입력한 뒤 "
            "00_environment_check.ipynb를 먼저 실행하세요."
        )
    return settings.openai_api_key


def get_chat_model(temperature: float = 0.0):
    """대화형 모델 인스턴스를 생성하는 공장 함수다."""
    from langchain_openai import ChatOpenAI

    settings = get_settings()
    try:
        return ChatOpenAI(
            model=settings.chat_model,
            temperature=temperature,
            api_key=_require_api_key(),
        )
    except Exception as exc:  # noqa: BLE001 - 학습자에게 원인을 함께 제공
        raise RuntimeError(
            f"Chat 모델 초기화에 실패했습니다 (model={settings.chat_model}). "
            f"원인: {exc}"
        ) from exc


def get_embedding_model():
    """벡터 임베딩 모델 인스턴스를 생성하는 공장 함수다."""
    from langchain_openai import OpenAIEmbeddings

    settings = get_settings()
    try:
        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=_require_api_key(),
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Embedding 모델 초기화에 실패했습니다 (model={settings.embedding_model}). "
            f"원인: {exc}"
        ) from exc
