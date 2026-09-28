# 5.modular_rag_manual/common/config.py

from pathlib import Path
import os

from dotenv import load_dotenv


# ------------------------------------------------------------
# Project Path
# ------------------------------------------------------------
# 현재 파일 위치:
# 3.modular_rag_manual/common/config.py
# parents[3] = 프로젝트 루트

# parents[1] = 6.modular_rag_manual/ (모듈 루트)
# parents[2] = langgraph_modular_rag_v1/ (레포 루트, .env 파일 위치)
BASE_DIR = Path(__file__).resolve().parents[1]
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(dotenv_path=_PROJECT_ROOT / ".env", override=True)

DATA_DIR = BASE_DIR / "data"
PDF_PATH = DATA_DIR / "공직자_민원응대_핵심_매뉴얼.pdf"

# ------------------------------------------------------------
# Environment
# ------------------------------------------------------------
ENV_PATH = _PROJECT_ROOT / ".env"
# override=True: 시스템 환경변수보다 .env 파일 값을 우선 적용
load_dotenv(dotenv_path=ENV_PATH, override=True)


# ------------------------------------------------------------
# Document
# ------------------------------------------------------------
# .env에 PDF_PATH를 지정하면 다른 파일로 교체 가능, 없으면 기본 경로 사용
PDF_PATH = Path(
    os.getenv(
        "PDF_PATH",
        str(DATA_DIR / "공직자_민원응대_핵심_매뉴얼.pdf"),
    )
)

# 노트북 1-2에서 생성한 전처리 Markdown page chunk 파일
CHUNK_MD_PATH = Path(
    os.getenv(
        "CHUNK_MD_PATH",
        str(DATA_DIR / "공직자_민원응대_핵심_매뉴얼_rag_page_chunks.md"),
    )
)


# ------------------------------------------------------------
# Qdrant
# ------------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
QDRANT_TIMEOUT = int(os.getenv("QDRANT_TIMEOUT", "10"))

# "medium" 접미사: 중간 크기의 청크(chunk) 전략으로 구성된 컬렉션임을 표시
COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "civil_complaint_manual_medium",
)


# ------------------------------------------------------------
# Model (NVIDIA OpenAI 호환 엔드포인트)
# ------------------------------------------------------------
NVIDIA_API_KEY = os.getenv("nvidiaapi_key")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "nvidia/nemotron-3-embed-1b",
)

LLM_MODEL = os.getenv(
    "LLM_MODEL",
    "z-ai/glm-5.3-flash",
)


# ------------------------------------------------------------
# Retrieval
# ------------------------------------------------------------
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))


# ------------------------------------------------------------
# Validation
# ------------------------------------------------------------
def validate_common_settings() -> None:
    """공통 설정값을 점검한다."""

    if not os.getenv("nvidiaapi_key"):
        raise EnvironmentError(
            "nvidiaapi_key가 설정되어 있지 않습니다. "
            ".env 파일에 nvidiaapi_key를 추가하세요."
        )

    if not COLLECTION_NAME:
        raise ValueError("COLLECTION_NAME이 비어 있습니다.")

    if not EMBEDDING_MODEL:
        raise ValueError("EMBEDDING_MODEL이 비어 있습니다.")

    if not QDRANT_URL:
        raise ValueError("QDRANT_URL이 비어 있습니다.")