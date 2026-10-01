"""환경변수와 실습 공통 Settings.

모델 인스턴스 생성은 :mod:`agentic_ai.models`가 담당한다. 이전 Notebook과의
호환성을 위해 이 모듈에도 모델 Factory wrapper를 유지한다.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

from .paths import ENV_FILE

# 공통 .env 파일을 읽어 노트북과 스크립트가 같은 설정을 사용하도록 맞춘다.
# OS에 남아 있는 이전 과정의 환경변수가 실습 설정을 덮지 않도록 한다.
load_dotenv(ENV_FILE, override=True)


# 실습에서 자주 쓰는 설정 값을 타입 안정성 있게 묶어 둔 데이터 클래스다.
@dataclass(frozen=True)
class Settings:
    openai_api_key: str | None
    chat_model: str
    embedding_model: str

    @property
    def api_key_configured(self) -> bool:
        """API Key가 비어 있지 않으면 True를 반환한다."""
        return bool(self.openai_api_key)


def get_settings() -> Settings:
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4.1-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
    )


def get_chat_model(temperature: float = 0.0):
    """이전 Import 경로를 위한 호환 wrapper."""
    from .models import get_chat_model as create_chat_model

    return create_chat_model(temperature=temperature)


def get_embedding_model():
    """이전 Import 경로를 위한 호환 wrapper."""
    from .models import get_embedding_model as create_embedding_model

    return create_embedding_model()
