## LoRA와 PEFT의 관계

핵심부터 정리하면 다음과 같다.

> **PEFT는 효율적으로 파인튜닝하는 방법들의 큰 범주이고, LoRA는 PEFT에 속하는 대표적인 방법이다.**

```text
Fine-Tuning
│
├── Full Fine-Tuning (FFT)
│   └── 모델 전체 파라미터 학습
│
└── PEFT
    │  Parameter-Efficient Fine-Tuning
    │
    ├── LoRA          ← 가장 많이 사용하는 방식
    ├── Prefix Tuning
    ├── Prompt Tuning
    └── Adapter 계열
```

### 1. PEFT란?

**PEFT(Parameter-Efficient Fine-Tuning)**는 거대한 LLM의 모든 파라미터를 학습하지 않고, **일부 파라미터만 학습하여 모델을 특정 작업에 적응시키는 접근법**이다.

예를 들어 원본 모델이 70억 개 파라미터를 가진다고 하면,

```text
Full Fine-Tuning

7B Model
┌─────────────────────────────┐
│ W1  W2  W3  W4 ... Wn      │
│ ↓   ↓   ↓   ↓       ↓      │
│ 모두 학습                   │
└─────────────────────────────┘
```

PEFT에서는 대부분의 원본 파라미터를 고정한다.

```text
PEFT

7B Model
┌─────────────────────────────┐
│ W1  W2  W3  W4 ... Wn      │
│ 🔒  🔒  🔒  🔒      🔒     │
│ 원본 파라미터 동결          │
└─────────────────────────────┘
             +
       일부 파라미터만 학습
```

따라서 GPU 메모리와 저장 공간을 크게 절감할 수 있다.

---

## 2. LoRA란?

**LoRA(Low-Rank Adaptation)**는 PEFT를 구현하는 대표적인 기법이다.

핵심 아이디어는 기존 가중치 `W` 자체를 변경하지 않고, 작은 행렬 `A`, `B`를 추가하여 이 행렬만 학습하는 것이다.

기존 Linear Layer가

\[
y = Wx
\]

라면 LoRA에서는 다음과 같이 생각할 수 있다.

\[
y = Wx + \Delta Wx
\]

그리고

\[
\Delta W = BA
\]

로 만든다.

따라서

\[
\boxed{y = Wx + BAx}
\]

가 된다.

구조로 보면 다음과 같다.

```text
                    입력 x
                      │
          ┌───────────┴───────────┐
          │                       │
          ▼                       ▼
    기존 Weight W             LoRA A
       Frozen                    │
          │                      ▼
          │                   LoRA B
          │                      │
          ▼                      ▼
         Wx                     BAx
          │                      │
          └──────────┬───────────┘
                     ▼
                  Wx + BAx
                     │
                     ▼
                   출력 y
```

여기서 중요한 것은 다음과 같다.

```text
W       → 학습하지 않음
A       → 학습
B       → 학습
```

즉,

```python
W.requires_grad = False

A.requires_grad = True
B.requires_grad = True
```

라고 이해하면 된다.

---

## 3. 왜 `Low-Rank`인가?

예를 들어 원래 Linear Layer가 다음과 같다고 하자.

```python
nn.Linear(4096, 4096)
```

원래 가중치 `W`의 크기는

\[
4096 \times 4096
\]

이므로 약 **1,677만 개**의 파라미터가 있다.

LoRA에서 `r=8`을 사용하면 이를 작은 두 행렬로 분해한다.

```text
W
4096 × 4096
약 16.8M parameters

대신

A : 8 × 4096
B : 4096 × 8
```

학습 파라미터는

\[
8 \times 4096 + 4096 \times 8
\]

즉,

\[
65,536
\]

개만 필요하다.

대략적으로 비교하면

```text
Full Fine-Tuning
16,777,216 parameters 학습

          ↓

LoRA (r=8)
65,536 parameters 학습
```

이것이 LoRA가 메모리 효율적인 핵심 이유이다.

---

## 4. Hugging Face의 `peft`는 무엇인가?

여기서 용어를 구분하는 것이 중요하다.

**PEFT는 개념이면서 동시에 Hugging Face 라이브러리 이름이기도 하다.**

Python에서는 다음 패키지를 사용한다.

```python
from peft import LoraConfig, get_peft_model, TaskType
```

역할은 다음과 같다.

```text
peft
 │
 ├── LoraConfig
 │     └── LoRA 설정 정의
 │
 ├── get_peft_model()
 │     └── 기존 모델에 LoRA 구조 삽입
 │
 └── TaskType
       └── 모델 작업 유형 지정
```

예를 들어 LLM에 LoRA를 적용하면 다음과 같다.

```python
from peft import LoraConfig, get_peft_model, TaskType

# LoRA 설정
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,

    # LoRA를 적용할 Linear Layer
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj"
    ],

    bias="none",

    # Causal Language Model
    task_type=TaskType.CAUSAL_LM
)

# 기존 모델에 LoRA 적용
model = get_peft_model(
    model,
    lora_config
)
```

이때 모델 구조는 개념적으로 다음과 같이 바뀐다.

```text
기존 Transformer
       │
       ▼
Self-Attention
       │
       ├── q_proj ── W
       ├── k_proj ── W
       ├── v_proj ── W
       └── o_proj ── W

             ↓ get_peft_model()

LoRA 적용 Transformer
       │
       ▼
Self-Attention
       │
       ├── q_proj ── W + LoRA(A,B)
       ├── k_proj ── W + LoRA(A,B)
       ├── v_proj ── W + LoRA(A,B)
       └── o_proj ── W + LoRA(A,B)
```

원래 `W`는 그대로 두고 **LoRA A/B가 추가되는 것**이다.

---

## 5. `r`과 `lora_alpha`의 의미

LoRA에서 특히 중요한 하이퍼파라미터가 다음 두 개이다.

```python
LoraConfig(
    r=8,
    lora_alpha=16
)
```

### `r`

`r`은 LoRA 행렬의 **rank**, 즉 저차원 공간의 크기이다.

```text
원래 W

4096
 │
 ▼
┌────────────────────┐
│    4096 × 4096     │
└────────────────────┘


LoRA

4096
 │
 ▼
┌────────────┐
│ A          │
│ 8 × 4096   │
└─────┬──────┘
      │ r = 8
      ▼
┌────────────┐
│ B          │
│ 4096 × 8   │
└────────────┘
```

`r`이 커질수록 학습 가능한 파라미터가 증가하고 표현력도 증가하지만 메모리 사용량도 늘어난다.

### `lora_alpha`

LoRA 업데이트의 영향력을 조절하는 scaling 값이다.

일반적으로 LoRA의 업데이트는 개념적으로

\[
\Delta W = \frac{\alpha}{r}BA
\]

형태로 적용된다.

예를 들어

```python
r = 8
lora_alpha = 16
```

이면

\[
\frac{16}{8}=2
\]

가 된다.

---

## 6. LoRA와 QLoRA는 다르다

여기서 수업할 때 가장 많이 혼동하는 부분이 있다.

```text
PEFT
 └── LoRA
```

와

```text
Quantization + LoRA
        ↓
      QLoRA
```

는 구분해야 한다.

### LoRA

```text
Base Model
BF16 / FP16
     │
     ├── 기존 W : Frozen
     │
     └── LoRA A/B : Train
```

### QLoRA

```text
Base Model
4-bit Quantization
     │
     ├── 기존 W : 4bit + Frozen
     │
     └── LoRA A/B : Train
```

따라서 코드에서

```python
BitsAndBytesConfig(
    load_in_4bit=True
)
```

와 LoRA를 함께 사용하면 일반적으로 **QLoRA 방식**이라고 이해하면 된다.

---

## 7. 전체 관계



```text
                    Fine-Tuning
                         │
          ┌──────────────┴──────────────┐
          │                             │
          ▼                             ▼
 Full Fine-Tuning                     PEFT
          │                             │
  전체 Weight 학습            일부 파라미터만 학습
                                        │
                              ┌─────────┼─────────┐
                              │         │         │
                            LoRA     Prefix    Prompt
                              │
                              ▼
                       Low-Rank Matrix
                          A × B 학습
                              │
                    ┌─────────┴─────────┐
                    │                   │
                  LoRA                QLoRA
                    │                   │
              BF16/FP16 Base      4-bit Base
                    │                   │
                    └─────────┬─────────┘
                              ▼
                        LoRA A/B 학습
```

한 문장으로 정리하면 **PEFT는 "전체 모델을 학습하지 않고 효율적으로 파인튜닝하자"는 접근이고, LoRA는 "원본 가중치는 고정하고 작은 저랭크 행렬 A와 B만 학습하자"는 PEFT 구현 기법이며, QLoRA는 여기에 Base Model의 4bit 양자화를 결합한 방식**이다.