# external_adapter/python_sandbox.py

# 파이썬 코드의 구조를 분석(추상 구문 트리, AST)하기 위한 내장 모듈
import ast
# 표준 사칙연산 함수(더하기, 빼기 등)를 제공하는 내장 모듈
import operator


class SafePythonSandbox:
    """
    안전하게 기본 산술 표현식만 평가하는 샌드박스 엔진 클래스입니다.
    
    [보안 목적]
    파이썬의 eval() 함수는 __import__('os').system('rm -rf /')와 같은 
    위험한 임의 코드를 실행할 수 있는 심각한 보안 약점이 있습니다.
    이 클래스는 AST(Abstract Syntax Tree) 분석을 통해 오직 허용된 
    숫자 및 기본 사칙연산 문법만 골라내어 실행하므로 코드 주입 공격을 원천 차단합니다.
    """

    # 1. 실행을 허용할 연산자 노드 타입과 실제 연산 함수의 매핑 테이블 정의
    # ast 모듈의 노드 객체(Key) -> operator 모듈의 연산 함수(Value)
    ALLOWED_OPERATORS = {
        ast.Add: operator.add,        # 덧셈 (+)
        ast.Sub: operator.sub,        # 뺄셈 (-)
        ast.Mult: operator.mul,       # 곱셈 (*)
        ast.Div: operator.truediv,    # 나눗셈 (/)
        ast.USub: operator.neg,       # 단항 음수 연산자 (예: -3)
        ast.UAdd: operator.pos,       # 단항 양수 연산자 (예: +5)
    }

    def calculate(self, expression: str) -> float | int:
        """
        수식 문자열(예: '3+5', '(12 + 8) * 3')을 전달받아 안전하게 결과값을 반환합니다.

        :param expression: 연산할 수식 문자열
        :return: 정수(int) 또는 실수(float) 형태의 계산 결과
        :raises ValueError: 허용되지 않는 구문/연산자나 올바르지 않은 수식일 경우 발생
        """
        try:
            # ast.parse(): 수식 문자열을 구문 분석하여 AST(추상 구문 트리) 구조로 변환합니다.
            # mode='eval': 단순 표현식(Single Expression) 형태만 파싱하도록 지정합니다.
            #             (변수 할당, 함수 정의, for문 등의 문장 구문은 제한됨)
            node = ast.parse(expression, mode='eval')

            # 파싱된 AST 트리의 최상위 바디 노드(node.body)부터 재귀적으로 검증 및 평가를 시작합니다.
            return self._eval_node(node.body)

        except Exception as e:
            # 예외 발생 시 원래 에러 메시지를 포함하여 사용자 친화적인 ValueError로 재전환
            raise ValueError(f"올바르지 않거나 허용되지 않은 수식입니다: {e}")

    def _eval_node(self, node):
        """
        AST 트리의 각 노드를 재귀적으로 순회하면서 검증 및 연산을 수행하는 내부 메서드입니다.

        :param node: 현재 검사 중인 AST 노드 객체
        :return: 계산된 수치 값 (int 또는 float)
        """
        # [조건 1] 노드가 단순 상수(리터럴 값)인 경우 (예: 숫자 3, 5 등)
        if isinstance(node, ast.Constant):
            # 상수의 값이 정수(int) 또는 실수(float)인 경우에만 허용
            if isinstance(node.value, (int, float)):
                return node.value
            # 문자열("hello"), 불리언(True/False), None 등의 상수가 들어오면 거부
            raise ValueError("숫자 형식만 허용됩니다.")

        # [조건 2] 노드가 이항 연산(Binary Operation)인 경우 (예: 3 + 5, 10 * 2)
        elif isinstance(node, ast.BinOp):
            # 현재 연산자의 타입 추출 (예: ast.Add, ast.Mult 등)
            op_type = type(node.op)

            # ALLOWED_OPERATORS 딕셔너리에 정의되지 않은 연산자인 경우 예외 발생 (예: 비트 연산자, 거듭제곱 등)
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"허용되지 않은 연산자입니다: {op_type.__name__}")

            # 재귀 호출을 통해 좌측 피연산자(left)와 우측 피연산자(right)의 값을 구함
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)

            # 허용된 연산자 매핑 테이블에서 해당되는 operator 함수를 가져와 두 값에 적용 후 반환
            return self.ALLOWED_OPERATORS[op_type](left, right)

        # [조건 3] 노드가 단항 연산(Unary Operation)인 경우 (예: -3, +5)
        elif isinstance(node, ast.UnaryOp):
            # 연산자 타입 추출 (예: ast.USub)
            op_type = type(node.op)

            # 허용 여부 검증
            if op_type not in self.ALLOWED_OPERATORS:
                raise ValueError(f"허용되지 않은 연산자입니다: {op_type.__name__}")

            # 단항 연산의 대상 피연산자(operand) 값을 재귀적으로 계산
            operand = self._eval_node(node.operand)

            # 해당 단항 연산 함수(operator.neg 등) 적용 후 반환
            return self.ALLOWED_OPERATORS[op_type](operand)

        # [조건 4] 그 외 허용되지 않은 모든 AST 노드 타입 차단
        # (예: ast.Name(변수명), ast.Call(함수 호출), ast.Attribute(속성 접근) 등)
        else:
            raise ValueError("허용되지 않은 문법 요소가 포함되어 있습니다.")