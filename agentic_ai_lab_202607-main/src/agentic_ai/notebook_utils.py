"""Notebook에서 반복 사용하는 환경 표시와 Graph 시각화 함수."""

from __future__ import annotations

import importlib.metadata
import sys
from typing import Any

from .config import Settings, get_settings
from .paths import DATA_DIR, OUTPUT_DIR, PROJECT_ROOT, ensure_output_dirs


CORE_PACKAGES = (
    "langchain",
    "langgraph",
    "langchain-openai",
    "langchain-chroma",
    "pydantic",
)


# 노트북 환경 확인용 패키지 버전 조회 함수다.
def package_versions(packages: tuple[str, ...] = CORE_PACKAGES) -> dict[str, str]:
    """설치된 핵심 패키지 버전을 반환한다."""
    versions: dict[str, str] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "설치되지 않음"
    return versions


# 학습자에게 필요한 환경 정보를 한눈에 보여주는 헬퍼다.
def print_environment_summary(
    settings: Settings | None = None,
    *,   # 여기서부터 키워드 전용 인자라는 구분의 의미다.
    needs_chat_model: bool = False,
    needs_embedding_model: bool = False,
) -> None:
    """학습자가 확인할 공통 환경 정보를 짧게 출력한다."""
    settings = settings or get_settings()
    ensure_output_dirs()

    print("[환경 설정 확인]")
    print(f"- 프로젝트: {PROJECT_ROOT}")
    print(f"- 데이터: {DATA_DIR}")
    print(f"- 출력: {OUTPUT_DIR}")

    if needs_chat_model or needs_embedding_model:
        status = "설정됨" if settings.api_key_configured else "설정되지 않음"
        print(f"- OPENAI_API_KEY: {status}")
    if needs_chat_model:
        print(f"- Chat Model: {settings.chat_model}")
    if needs_embedding_model:
        print(f"- Embedding Model: {settings.embedding_model}")

    if (needs_chat_model or needs_embedding_model) and not settings.api_key_configured:
        print("  → .env를 설정한 뒤 00_environment_check.ipynb를 먼저 실행하세요.")


def environment_report() -> dict[str, Any]:
    """00 환경 점검 Notebook에서 사용할 구조화된 점검 결과를 반환한다."""
    settings = get_settings()
    return {
        "python": sys.version.split()[0],
        "project_root": str(PROJECT_ROOT),
        "data_dir_exists": DATA_DIR.is_dir(),
        "output_dir_exists": OUTPUT_DIR.is_dir(),
        "env_file_exists": (PROJECT_ROOT / ".env").is_file(),
        "api_key_configured": settings.api_key_configured,
        "chat_model": settings.chat_model,
        "embedding_model": settings.embedding_model,
        "packages": package_versions(),
    }


def show_graph(graph: Any) -> None:
    """LangGraph의 Mermaid 그래프를 노트북 출력 영역에 표시한다."""
    from IPython.display import Image, display

    try:
        display(Image(graph.get_graph().draw_mermaid_png()))
    except Exception as e:
        print("show_graph(graph) 실패:", e)
