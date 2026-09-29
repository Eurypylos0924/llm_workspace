# app/core/settings.py

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict

# PROJECT_ROOT 정의 (settings.py 위치 기준 3단계 상위 디렉터리: my_mcp_project)
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    app_name: str = "MCP Management Portal"
    app_host: str = "127.0.0.1"
    app_port: int = 8000
    # NVIDIA OpenAI 호환 엔드포인트 (.env의 nvidiaapi_key 사용)
    nvidiaapi_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    llm_model: str = "z-ai/glm-5.3-flash"
    embedding_model: str = "nvidia/nemotron-3-embed-1b"
    tavily_api_key: str = ""

    # Pydantic v2 / pydantic-settings 표준 설정
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    return Settings()