"""프로젝트와 데이터·출력 경로를 한 곳에서 관리한다."""

from __future__ import annotations

from pathlib import Path


# 프로젝트 루트와 데이터/출력 폴더를 기준 경로로 한 번에 관리한다.
PACKAGE_DIR = Path(__file__).resolve().parent
SRC_DIR = PACKAGE_DIR.parent
PROJECT_ROOT = SRC_DIR.parent

ENV_FILE = PROJECT_ROOT / ".env"
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
LOG_DIR = OUTPUT_DIR / "logs"
RETRIEVAL_DIR = OUTPUT_DIR / "retrieval"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
VECTORSTORE_DIR = OUTPUT_DIR / "vectorstore"
CHROMA_DIR = VECTORSTORE_DIR / "chroma"


def ensure_output_dirs() -> None:
    """실습 실행 중 생성되는 결과물 디렉터리를 미리 준비한다."""
    for directory in (
        OUTPUT_DIR,
        LOG_DIR,
        RETRIEVAL_DIR,
        CHECKPOINT_DIR,
        CHROMA_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


def data_path(*parts: str, must_exist: bool = False) -> Path:
    """data 하위 경로를 만들고 필요하면 존재 여부를 확인한다."""
    path = DATA_DIR.joinpath(*parts)
    if must_exist and not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {path}")
    return path


def output_path(*parts: str, create_parent: bool = True) -> Path:
    """outputs 하위 경로를 반환하고 기본적으로 상위 폴더를 생성한다."""
    path = OUTPUT_DIR.joinpath(*parts)
    if create_parent:
        path.parent.mkdir(parents=True, exist_ok=True)
    return path
