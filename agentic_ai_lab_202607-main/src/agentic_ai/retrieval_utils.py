"""Chroma Collection 생성과 재사용을 위한 공통 Helper."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document

from .models import get_embedding_model
from .paths import CHROMA_DIR


# Chroma Vector Store를 재사용 가능한 형태로 열고, 없으면 새로 만든다.
def get_chroma_store(
    collection_name: str,
    *,
    embedding_model: Any | None = None,
    persist_directory: str | Path = CHROMA_DIR,
) -> Chroma:
    """디스크 기반 Chroma Collection을 열거나 새로 만든다."""
    directory = Path(persist_directory)
    directory.mkdir(parents=True, exist_ok=True)
    return Chroma(
        collection_name=collection_name,
        embedding_function=embedding_model or get_embedding_model(),
        persist_directory=str(directory),
        collection_metadata={"hnsw:space": "cosine"},
    )


# Collection이 비어 있는 경우에만 문서를 추가해 중복 삽입을 방지한다.
def add_documents_if_empty(
    vector_store: Chroma,
    documents: Sequence[Document],
    *,
    ids: Sequence[str],
) -> tuple[bool, int]:
    """Collection이 비어 있을 때만 문서를 추가한다.

    반환값은 ``(추가 여부, 현재 문서 수)``이다.
    """
    existing_ids = vector_store.get(include=[])["ids"]
    if existing_ids:
        return False, len(existing_ids)

    vector_store.add_documents(list(documents), ids=list(ids))
    return True, len(documents)
