# src/modular_rag_manual/common/embeddings.py

from functools import lru_cache

from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings

from config import EMBEDDING_MODEL, NVIDIA_API_KEY


# maxsize=1: 인자 없는 함수이므로 인스턴스를 1개만 캐싱 (싱글톤 효과)
# NVIDIA API 클라이언트를 매 호출마다 새로 생성하지 않기 위함
@lru_cache(maxsize=1)
def get_embeddings() -> NVIDIAEmbeddings:
    """프로젝트 공통 임베딩 모델을 생성한다."""

    return NVIDIAEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=NVIDIA_API_KEY,
    )


# get_embeddings()를 직접 호출하지 않는 레거시 코드를 위한 모듈 수준 별칭
embeddings = get_embeddings()