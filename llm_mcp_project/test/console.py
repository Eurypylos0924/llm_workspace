# test/console.py
from mcp_server.server import multiply

# 파이썬 함수처럼 직접 인자 전달하여 호출
result = multiply(4.5, 3.0)
print(f"곱셈 결과: {result}")  # 출력: 13.5