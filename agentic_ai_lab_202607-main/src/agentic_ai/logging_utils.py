"""실행 상태 출력과 실행 로그 저장을 위한 공통 유틸리티."""

from __future__ import annotations

import json
import csv
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def print_state(title: str, state: dict[str, Any]) -> None:
    """State 또는 실행 결과를 사람이 읽기 쉬운 형태로 출력한다."""
    print(f"\n=== {title} ===")
    for key, value in state.items():
        print(f"- {key}: {value}")


def print_steps(steps: list[dict[str, Any]]) -> None:
    """단계별 실행 기록을 순서대로 출력한다."""
    for i, step in enumerate(steps, start=1):
        print(f"\n[Step {i}]")
        for key, value in step.items():
            print(f"  {key}: {value}")


def create_run_log(case_id: str, thread_id: str = "thread-000") -> dict[str, Any]:
    """에이전트 실행 추적을 위해 공통 로그 스키마를 먼저 구성한다."""
    return {
        "case_id": case_id,
        "thread_id": thread_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "executed_nodes": [],
        "selected_tools": [],
        "tool_arguments": [],
        "tool_results": [],
        "retrieval_results": [],
        "retry_count": 0,
        "final_status": "running",
        "error": None,
    }


def save_log(log: dict[str, Any], output_path: str | Path) -> Path:
    """로그를 JSON 형식으로 저장해 디버깅과 재현에 사용할 수 있게 한다."""
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(log, f, ensure_ascii=False, indent=2)
    return path


def load_jsonl(file_path: str | Path) -> list[dict[str, Any]]:
    """JSONL 파일을 읽어 딕셔너리 리스트로 반환한다."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"데이터 파일을 찾을 수 없습니다: {file_path}")
    records = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_csv(records: list[dict[str, Any]], output_path: str | Path) -> Path:
    """딕셔너리 목록을 UTF-8 CSV로 저장한다."""
    if not records:
        raise ValueError("CSV로 저장할 records가 비어 있습니다.")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(records[0].keys())
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    return path
