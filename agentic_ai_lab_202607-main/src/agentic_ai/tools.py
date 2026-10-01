"""Tool Calling 실습에 사용하는 공통 Tool 함수 모음.

각 함수는 LLM이 아니라 애플리케이션이 직접 실행하는 일반 Python 함수이다.
LLM은 어떤 함수를 어떤 인자로 호출할지 제안할 뿐이다.
"""

from __future__ import annotations

import ast
import operator
from pathlib import Path
from typing import Any

_ALLOWED_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_ALLOWED_UNARY_OPS = {
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}


# 안전한 수식 평가를 위해 허용된 연산만 처리한다.
def _eval_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _ALLOWED_BIN_OPS:
        left = _eval_node(node.left)
        right = _eval_node(node.right)
        return _ALLOWED_BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _ALLOWED_UNARY_OPS:
        return _ALLOWED_UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("허용되지 않은 수식 요소가 포함되어 있습니다.")


def calculate(expression: str) -> int | float:
    """사칙연산 수식 문자열을 계산하여 숫자를 반환한다.

    예: calculate("25 + 17") -> 42
    """
    try:
        tree = ast.parse(expression, mode="eval")
        return _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise ValueError("0으로 나눌 수 없습니다.") from exc
    except (SyntaxError, ValueError) as exc:
        raise ValueError(f"수식을 계산할 수 없습니다: '{expression}' ({exc})") from exc


def search_keyword(text: str, keyword: str) -> dict[str, Any]:
    """텍스트 안에서 키워드의 등장 여부와 위치를 반환한다."""
    if not keyword:
        raise ValueError("keyword는 빈 문자열일 수 없습니다.")
    index = text.find(keyword)
    return {"keyword": keyword, "found": index != -1, "position": index}


def load_document(file_path: str) -> str:
    """지정한 경로의 문서를 읽어 본문 문자열을 반환한다."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"문서를 찾을 수 없습니다: {file_path}")
    return path.read_text(encoding="utf-8")


def check_required_fields(data: dict[str, Any], required: list[str]) -> list[str]:
    """필수 필드 중 값이 없거나 비어 있는 필드 목록을 반환한다."""
    return [
        field
        for field in required
        if field not in data or data[field] in (None, "")
    ]


# LLM이 호출할 수 있는 실제 Tool 함수들을 등록해 둔다.
TOOL_REGISTRY = {
    "calculate": calculate,
    "search_keyword": search_keyword,
    "load_document": load_document,
    "check_required_fields": check_required_fields,
}
