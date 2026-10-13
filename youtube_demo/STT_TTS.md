# STT·TTS 음성 인공지능: 원리, 모델, 구현 및 평가


## 학습 목표

1. STT와 TTS의 입출력과 기술적 차이를 설명한다.
2. 음성 파형·샘플링·스펙트로그램과 음성 모델의 관계를 이해한다.
3. STT의 전처리·인코딩·디코딩 과정과 대표 모델을 비교한다.
4. TTS의 정규화·G2P·음향 모델·보코더 구조를 설명한다.
5. Python으로 음성 인식과 음성 합성을 구현하고 결과를 평가한다.
6. STT–LLM–TTS를 연결한 음성 질의응답 시스템을 설계한다.

---


## 주요 구성 내용

| 구분            | 수록 내용                                            |
| ------------- | ------------------------------------------------ |
| 1. STT·TTS 개요 | 정의, 특징, 활용 분야, 비교                                |
| 2. 음성 신호 기초   | Waveform, Sampling, Mel Spectrogram              |
| 3. STT 처리 원리  | 전처리, Encoder, Decoder, CTC                       |
| 4. STT 모델     | Whisper, wav2vec 2.0, Conformer                  |
| 5. TTS 처리 원리  | Text Normalization, G2P, Acoustic Model, Vocoder |
| 6. TTS 모델     | Tacotron, FastSpeech, VITS, XTTS                 |
| 7. 성능 평가      | WER, CER, MOS, RTF                               |
| 8. Python 실습  | Whisper STT, 한국어 VITS TTS                        |
| 9. 시스템 통합     | STT → LLM → TTS, FastAPI 구조                      |
| 10. 프로젝트      | 미니 프로젝트, 평가 기준                       |




# 1. STT와 TTS 개요

## 1.1 STT(Speech-to-Text) 

STT는 음성 신호를 분석하여 사람이 발화한 내용을 텍스트로 변환하는 기술이다. ASR(Automatic Speech Recognition, 자동 음성 인식)이라고도 한다.

```text
마이크 / 음성 파일 → 음성 전처리 → 음성 인식 모델 → 디코딩 → 텍스트
      "오늘 날씨 알려줘"                         "오늘 날씨 알려줘"
```

주요 활용 분야는 회의록 자동 작성, 동영상 자막, 콜센터 상담 분석, 음성 명령, 접근성 지원 등이다.

## 1.2 TTS(Text-to-Speech) 
TTS는 텍스트를 사람이 들을 수 있는 음성으로 합성하는 기술이다. 사용자가 입력한 문장이나 LLM이 생성한 답변을 자연스러운 음성으로 출력한다.

```text
텍스트 "안녕하세요" → TTS 엔진 → 디지털 음성 파형 → 스피커
```

대표 활용 분야는 스마트 스피커, 내비게이션, 전자책 낭독, 교육, ARS, AI 안내 및 접근성 서비스이다.

## 1.3 비교

| 구분 | STT | TTS |
|---|---|---|
| 정식 명칭 | Speech-to-Text | Text-to-Speech |
| 목적 | 음성 내용 인식 | 텍스트의 음성 합성 |
| 입력 | 음성 파형 | 텍스트·언어 정보 |
| 출력 | 전사 텍스트 | 음성 파형 |
| 주요 과제 | 잡음, 억양, 발화 속도, 고유명사 | 발음, 억양, 리듬, 화자 특성 |
| 대표 모델 | Whisper, wav2vec 2.0, Conformer, CTC/RNN-T 계열 | Tacotron 2, FastSpeech 2, VITS, XTTS 계열 |
| 주요 평가 | WER, CER, 지연시간 | MOS, 자연스러움, 발음 정확도, 지연시간 |

**주의:** STT와 TTS는 입력·출력 방향이 반대지만, 동일한 모델을 단순히 역방향 실행하는 관계는 아니다. 서로 다른 학습 목표와 구조를 사용한다.

---

# 2. 음성 신호의 기초

## 2.1 파형(Waveform)

음성은 시간에 따른 공기압 변화를 디지털 값으로 표본화한 신호이다. 단일 채널 음성은 일반적으로 `audio.shape == (num_samples,)` 형태의 배열로 표현할 수 있다.

- **샘플링 레이트(Sample Rate)**: 1초당 표본 수. 16 kHz는 1초에 16,000개 표본을 의미한다.
- **비트 깊이(Bit Depth)**: 표본당 진폭 표현 정밀도. 예: PCM 16-bit.
- **채널(Channel)**: 모노(1), 스테레오(2) 등.
- **길이(Duration)**: `표본 수 / 샘플링 레이트`.

예를 들어 16 kHz 모노 음성 5초의 표본 수는 `16,000 × 5 = 80,000`개이다.

## 2.2 스펙트로그램과 Mel 스펙트로그램

스펙트로그램은 시간에 따라 변화하는 주파수 성분을 나타낸다. Mel 스펙트로그램은 인간 청각의 주파수 지각 특성을 반영한 Mel 필터뱅크를 적용한 표현이다.

```text
Waveform → STFT → Magnitude/Power Spectrogram → Mel Filterbank → Mel Spectrogram
```

STT에서는 모델에 따라 원시 파형 또는 로그 Mel 특징을 입력받는다. TTS에서는 음향 모델의 중간 출력으로 Mel 스펙트로그램을 사용하는 구조가 널리 알려져 있으나, 모든 현대 TTS가 이 중간 표현을 반드시 거치는 것은 아니다.

## 2.3 음성 전처리

1. 채널 변환: 필요하면 스테레오를 모노로 변환한다.
2. 리샘플링: 모델이 요구하는 샘플링 레이트로 변환한다. **숫자만 변경하면 안 되고 실제 신호를 리샘플링해야 한다.**
3. 음량 조정: 클리핑을 피하면서 레벨을 정규화한다.
4. VAD(Voice Activity Detection): 음성 구간을 탐지해 긴 무음 구간을 제외한다.
5. 분할: 긴 음성은 문맥과 겹침 구간을 고려하여 청크로 처리한다.
6. 선택적 잡음 감소: 과도한 제거는 음성 정보를 손상할 수 있다.

---

# 3. STT 처리 원리

## 3.1 전체 파이프라인

```text
[마이크 또는 음성 파일]
          ↓
[샘플링·채널·음량 정리]
          ↓
[VAD / 음성 구간 분할(선택)]
          ↓
[특징 추출: 파형 또는 로그 Mel]
          ↓
[음성 인코더: 음향·언어 관련 표현]
          ↓
[디코더: CTC / Transducer / Attention 등]
          ↓
[후처리: 띄어쓰기·구두점·용어 보정]
          ↓
[전사 텍스트 및 선택적 타임스탬프]
```

### 음성 인코더

음성의 시간적 패턴과 발음 정보를 신경망 표현으로 변환한다. CNN, Transformer, Conformer 등의 구조를 사용할 수 있다.

### 디코더

인코더 출력을 글자, 서브워드, 토큰 시퀀스로 변환한다. 모델마다 디코딩 방식이 다르다.

- **CTC(Connectionist Temporal Classification)**: 입력 프레임과 출력 문자 간 정확한 정렬을 제공하지 않아도 학습할 수 있도록 설계된 목적 함수이다. 반복 토큰과 blank 토큰을 정리해 최종 문자열을 만든다.
- **RNN-T / Transducer**: 입력 음성과 이전 출력 토큰을 함께 고려하며 스트리밍 음성 인식에 적합한 계열이다.
- **Attention Encoder–Decoder**: 인코더 표현을 참조하여 토큰을 순차적으로 생성한다. Whisper가 대표적인 사례이다.

## 3.2 대표 STT 모델

| 모델·계열 | 핵심 특징 | 적합한 상황 | 유의점 |
|---|---|---|---|
| wav2vec 2.0 | 원시 파형 기반 자기지도 사전학습 | 도메인 데이터로 추가 학습 | 실제 전사는 CTC 헤드 등 추가 구성이 필요할 수 있음 |
| Conformer | CNN과 Transformer를 결합 | 고정밀 음성 인식 인코더 | 독립적인 단일 전사 모델명이 아니라 아키텍처 계열 |
| Whisper | 다국어 음성 인식·번역용 Encoder–Decoder | 파일 기반 전사, 다국어 실습 | 긴 무음·배경음에서 환각 전사 가능 |
| CTC 기반 모델 | 프레임 정렬을 단순화 | 비교적 단순한 디코딩 | 문맥 복원 및 구두점은 별도 처리할 수 있음 |
| RNN-T 기반 모델 | 온라인·스트리밍 인식에 적합 | 실시간 음성 입력 | 배포 및 디코더 최적화가 중요 |

모델 성능은 언어, 녹음 환경, 데이터셋, 하드웨어 및 구현 방식에 따라 달라진다. 단일 모델을 모든 환경에서 최상이라고 단정할 수 없다.

## 3.3 Whisper 구조

```text
음성 파형 → 로그 Mel 스펙트로그램 → Audio Encoder
                                         ↓
                       Text Decoder(이전 토큰 + 인코더 정보)
                                         ↓
                                  전사 또는 번역 텍스트
```

Whisper는 음성 인식 및 음성 번역을 지원한다. 일반 전사와 번역은 태스크 설정이 다르며, 언어와 모델 크기에 따라 지원 범위가 달라진다.

## 3.4 STT의 주요 오류

- **치환(Substitution)**: 정답 `사과` → 예측 `사고`.
- **삭제(Deletion)**: 정답 `오늘 날씨` → 예측 `날씨`.
- **삽입(Insertion)**: 정답 `안녕하세요` → 예측 `네 안녕하세요`.
- **환각(Hallucination)**: 무음이나 비음성 구간에서 존재하지 않는 문장을 생성.
- **화자 혼동**: 복수 화자의 발화를 잘못 합치거나 분리.

**화자 분리(Diarization)**는 누가 언제 말했는지 구분하는 작업이며, 음성을 텍스트로 바꾸는 STT 자체와 구별한다.

---

# 4. TTS 처리 원리
## 4.1 전통적인 신경망 TTS 파이프라인

```text
텍스트
  ↓
텍스트 정규화(Text Normalization)
  ↓
문자→발음 변환(G2P)
  ↓
음향 모델(Acoustic Model)
  ↓
Mel Spectrogram
  ↓
신경망 보코더(Neural Vocoder)
  ↓
Waveform → 오디오 파일 또는 스피커
```

이는 **대표적인 2단계 TTS 구조**이다. VITS 등 통합형 모델이나 토큰·코덱 기반 생성 모델은 별도 Mel 스펙트로그램·보코더 단계를 동일한 방식으로 노출하지 않을 수 있다.

## 4.2 텍스트 정규화

숫자, 날짜, 단위, 기호, 약어 등을 문맥에 맞는 읽기 형태로 변환한다.

| 원문 | 가능한 읽기 예시 | 고려사항 |
|---|---|---|
| `2026년` | 이천이십육 년 | 연도 읽기 |
| `3kg` | 삼 킬로그램 | 단위 표기 |
| `AI` | 에이아이 | 약어 발음 |
| `10:30` | 열 시 삼십 분 | 시각인지 비율인지 문맥 확인 |
| `1,000원` | 천 원 | 금액·수량 표현 |

정규화 결과는 문맥과 도메인에 따라 달라질 수 있다.

## 4.3 G2P(Grapheme-to-Phoneme)

표기된 글자를 실제 발음에 가까운 음소열로 변환한다. 한국어에서는 연음, 된소리되기, 비음화, 구개음화, 받침 발음 등이 중요하다.

- `학교` → `[학꾜]`
- `같이` → `[가치]`

G2P는 한국어 TTS의 발음 품질에 큰 영향을 준다. 다만 일부 모델은 별도의 명시적 G2P 없이 문자·토큰을 입력받아 학습한다.

## 4.4 음향 모델과 Mel 스펙트로그램

음향 모델은 텍스트·발음 정보에서 음성의 시간적·주파수적 특성을 예측한다. 길이(Duration), 음높이(Pitch), 에너지(Energy), 운율(Prosody)이 중요한 요소이다.

- **Duration**: 음소·단어가 얼마나 길게 발음되는지.
- **Pitch(F0)**: 기본 주파수와 억양의 변화.
- **Energy**: 음성의 상대적 강도.
- **Prosody**: 리듬, 강세, 억양, 휴지 등.

## 4.5 Neural Vocoder

음향 특징을 최종 음성 파형으로 변환한다. 원문에서 제시한 주요 모델은 WaveNet, WaveGlow, HiFi-GAN, Parallel WaveGAN, UnivNet, BigVGAN이다. 모델에 따라 음질, 속도, 학습 복잡도가 다르다.

## 4.6 대표 TTS 모델
| 모델 | 핵심 구조 | 특징 |
|---|---|---|
| Tacotron | Attention Encoder–Decoder | 텍스트에서 음향 특징 생성, 순차 디코딩 |
| Tacotron 2 | 음향 모델 + 신경망 보코더 | 고품질 음성 합성의 대표적인 2단계 방식 |
| FastSpeech | 비자기회귀 Transformer, Length Regulator | 병렬 생성으로 추론 속도 개선 |
| FastSpeech 2 | Duration·Pitch·Energy 예측 | 운율 요소를 명시적으로 모델링 |
| VITS | 변분 추론 + 적대적 학습을 활용한 통합형 모델 | 음향 생성과 파형 합성을 통합 |
| XTTS 계열 | 다국어·화자 조건부 합성 | 음성 참조를 통한 화자 특성 반영 가능 |

**정확성 보완:** Tacotron 2의 보코더는 대표적인 구성에서 WaveNet을 사용했지만, 모든 Tacotron 2 구현이 WaveNet으로 고정되는 것은 아니다. 또한 최신 TTS를 모두 Transformer 기반이라고 일반화할 수 없다.

## 4.7 TTS 학습 절차

```text
텍스트–음성 쌍 데이터 수집
             ↓
텍스트·음성 정제 및 정규화
             ↓
정렬(Alignment) 학습 또는 추정
             ↓
음향 모델 / 통합형 모델 학습
             ↓
보코더 학습(분리형 모델인 경우)
             ↓
객관적·주관적 품질 평가
             ↓
모델 최적화 및 배포
```

원문은 텍스트–음성 쌍을 전제로 설명한다. 다만 최신 생성형 음성 모델의 사전학습·미세조정에는 다양한 데이터 구성과 학습 목표가 사용될 수 있다.

---

# 5. STT·TTS 학습 데이터 및 평가

## 5.1 데이터 설계

**STT 데이터**는 음성 파일과 정답 전사문이 기본이다. **TTS 데이터**는 텍스트와 그 텍스트를 발화한 음성의 대응 관계가 중요하다.

```text
STT 예시: {"audio": "speech_001.wav", "transcript": "안녕하세요"}
TTS 예시: {"text": "안녕하세요", "audio": "voice_001.wav", "speaker_id": "spk01"}
```

공통 품질 관리 항목은 녹음 품질, 음성·문장 정합성, 언어·방언 다양성, 개인정보 보호, 사용 권한 및 학습/검증/시험 분리이다. 화자 기반 일반화 평가에서는 동일 화자가 학습·시험 세트에 중복되지 않도록 분리하는 방식을 고려한다.

## 5.2 STT: WER와 CER

WER(Word Error Rate)은 단어 수준 편집 오류율이다.

\[
WER = \frac{S + D + I}{N}
\]

- `S`: 치환된 단어 수
- `D`: 삭제된 단어 수
- `I`: 삽입된 단어 수
- `N`: 정답 단어 수

CER(Character Error Rate)은 같은 원리를 문자 단위로 적용한다.

\[
CER = \frac{S_c + D_c + I_c}{N_c}
\]

한국어에서는 띄어쓰기·숫자·문장부호 정규화 정책에 따라 WER가 크게 달라질 수 있다. 따라서 평가 전 전처리 기준을 고정하고 원본 전사문도 보존해야 한다. WER/CER는 0에 가까울수록 좋지만, 삽입 오류가 많으면 1을 초과할 수도 있다.

## 5.3 TTS 평가

| 지표 | 평가 대상 | 해석 |
|---|---|---|
| MOS(Mean Opinion Score) | 청취자가 평가하는 음질·자연스러움 | 보통 1~5점 주관 평가, 설문 설계 필요 |
| CMOS | 두 음성의 상대적 선호 | 비교 평가 |
| 발음 정확도 | 잘못 읽은 단어·숫자·고유명사 | 도메인별 테스트 문장 필요 |
| 화자 유사도 | 목표 화자와 합성 음성의 특성 유사성 | 임베딩 유사도는 인간 평가와 동일하지 않음 |
| RTF(Real-Time Factor) | 처리 시간 / 음성 길이 | 일반적으로 1 미만이면 실시간보다 빠른 처리 |
| 최초 오디오 지연 | 요청 후 첫 오디오가 출력되기까지 시간 | 스트리밍 대화형 서비스에서 중요 |

**주의:** STT를 이용해 합성 음성을 재전사하는 방식은 TTS 발음 평가를 보조할 수 있지만, 자연스러움이나 감정 표현을 직접 평가하는 지표는 아니다.

---

# 6. 실습 환경 구성

## 6.1 Windows PowerShell

Python 3.11 또는 3.12 환경을 예시로 사용한다. 설치 전 Python·PyTorch·Transformers 간 버전 호환성을 확인해야 한다. 다음 코드는 교육용 CPU/GPU 공통 흐름이며, 모델 최초 다운로드에는 인터넷이 필요하다.

```powershell
# 프로젝트 폴더를 생성한다.
New-Item -ItemType Directory -Force -Path .\speech-ai
# 작업 폴더로 이동한다.
Set-Location .\speech-ai
# Python 가상환경을 생성한다.
py -3.12 -m venv .venv
# 가상환경을 활성화한다.
.\.venv\Scripts\Activate.ps1
# pip를 업데이트한다.
python -m pip install --upgrade pip
# 음성 인식·합성 실습 패키지를 설치한다.
python -m pip install torch transformers soundfile librosa jiwer sentencepiece
```

CUDA GPU에서 실행할 경우 PyTorch 공식 설치 안내에 따라 장치에 맞는 PyTorch 빌드를 별도로 설치한다. GPU가 없어도 실행은 가능하지만 속도가 느릴 수 있다.

## 6.2 장치 선택

```python
# PyTorch를 불러온다.
import torch
# CUDA → Apple MPS → CPU 순서로 사용 가능한 장치를 선택한다.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# 선택된 장치를 출력한다.
print("사용 장치:", device)
```

---

# 7. 실습 1: Whisper로 STT 구현

**목표:** 한국어 음성 파일 `sample_ko.wav`를 입력받아 텍스트를 생성한다.

**사전 준비:** 학습자가 발음한 한국어 문장을 녹음하여 프로젝트 폴더에 `sample_ko.wav`로 저장한다. Whisper의 `openai/whisper-tiny` 모델은 학습·시연용 소형 모델이므로 한국어 정확도는 제한적일 수 있다.

```python
# 파일 경로를 다루기 위한 Path를 가져온다.
from pathlib import Path
# 오디오를 읽고 필요 시 리샘플링하기 위해 librosa를 가져온다.
import librosa
# 모델을 로드하기 위해 Transformers 클래스를 가져온다.
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor
# PyTorch를 가져온다.
import torch
# CPU·GPU 장치를 자동으로 선택한다.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# 교육용 소형 Whisper 모델을 지정한다.
model_id = "openai/whisper-tiny"
# 모델과 일치하는 전처리기를 불러온다.
processor = AutoProcessor.from_pretrained(model_id)
# 사전학습된 Whisper 음성 인식 모델을 불러온다.
model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id).to(device)
# 추론 모드로 전환한다.
model.eval()
# 오디오 파일의 존재 여부를 확인한다.
audio_path = Path("sample_ko.wav")
# 입력 파일이 없으면 명확한 오류를 발생시킨다.
if not audio_path.exists():
    raise FileNotFoundError("sample_ko.wav 파일을 프로젝트 폴더에 준비해야 한다.")
# Whisper의 입력 규격인 16kHz 모노 파형으로 로드한다.
audio, sr = librosa.load(audio_path, sr=16000, mono=True)
# 빈 오디오 입력을 방지한다.
if len(audio) == 0:
    raise ValueError("음성 데이터가 비어 있다.")
# Whisper 입력 특징(log-Mel)을 계산한다.
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
# 입력 특징을 모델이 있는 장치로 이동한다.
input_features = inputs.input_features.to(device)
# 한국어 전사 태스크를 지정하는 프롬프트 토큰을 만든다.
forced_ids = processor.get_decoder_prompt_ids(language="korean", task="transcribe")
# 기울기 계산 없이 음성 인식을 수행한다.
with torch.inference_mode():
    predicted_ids = model.generate(input_features, forced_decoder_ids=forced_ids)
# 출력 토큰을 사람이 읽을 수 있는 문자열로 복원한다.
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
# 최종 전사 결과를 출력한다.
print("STT 결과:", transcription)
```

**버전 유의:** `get_decoder_prompt_ids`와 `forced_decoder_ids`의 사용 방식은 Transformers 버전에 따라 달라질 수 있다. 사용 중인 버전에서 경고 또는 오류가 발생하면 `generate(..., language="ko", task="transcribe")` 지원 여부를 확인한다. 긴 음성은 분할 및 타임스탬프 처리가 별도로 필요하다.

## 7.1 STT 결과 분석 과제

1. 동일한 문장을 조용한 장소와 소음이 있는 장소에서 녹음한다.
2. 모델을 동일하게 유지한 상태에서 결과를 비교한다.
3. 고유명사·숫자·영어 혼용 문장에 대한 오류를 기록한다.
4. 오인식 원인을 음향 환경·발음·언어 모델링 관점에서 분석한다.

---

# 8. 실습 2: STT 정확도 평가 [추가 보강]

```python
# WER와 CER 계산 함수를 불러온다.
from jiwer import wer, cer
# 사람이 확인한 정답 문장을 준비한다.
references = ["오늘 서울의 날씨를 알려줘", "인공지능 수업을 시작합니다"]
# STT 모델이 생성한 예측 문장을 준비한다.
predictions = ["오늘 서울 날씨를 알려줘", "인공지능 수업 시작합니다"]
# 단어 오류율을 계산한다.
word_error_rate = wer(references, predictions)
# 문자 오류율을 계산한다.
character_error_rate = cer(references, predictions)
# 평가 결과를 소수점 넷째 자리까지 출력한다.
print(f"WER: {word_error_rate:.4f}")
# 문자 오류율을 출력한다.
print(f"CER: {character_error_rate:.4f}")
```

**실무 해석:** 위 결과는 예제 문자열의 편집 오류율이며, 실제 음성 인식 모델의 일반화 성능을 의미하지 않는다. 공정한 모델 비교에는 동일한 테스트셋과 동일한 텍스트 정규화 규칙이 필요하다.

---

# 9. 실습 3: 한국어 TTS 구현 

다국어 TTS 모델 `facebook/mms-tts-kor`를 이용한다. 이 모델은 한국어 텍스트를 음성으로 합성하는 실습용 예시이다. 실제 배포 전 모델 라이선스, 품질, 운영 환경 적합성을 확인해야 한다.

```python
# 파형 파일 저장을 위해 soundfile을 가져온다.
import soundfile as sf
# PyTorch를 가져온다.
import torch
# VITS 모델과 토크나이저를 불러온다.
from transformers import VitsModel, AutoTokenizer
# 한국어 TTS 모델 식별자를 설정한다.
model_id = "facebook/mms-tts-kor"
# 사전학습된 한국어 토크나이저를 로드한다.
tokenizer = AutoTokenizer.from_pretrained(model_id)
# 사전학습된 VITS 모델을 로드한다.
model = VitsModel.from_pretrained(model_id)
# 사용 가능한 장치를 선택한다.
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
# 모델을 연산 장치로 이동한다.
model = model.to(device)
# 평가 모드로 전환한다.
model.eval()
# 합성할 한국어 문장을 지정한다.
text = "안녕하세요. 음성 인공지능 수업을 시작하겠습니다."
# 문장을 모델 입력 토큰으로 변환한다.
inputs = tokenizer(text, return_tensors="pt")
# 입력 텐서를 모델 장치로 이동한다.
inputs = {key: value.to(device) for key, value in inputs.items()}
# 추론 전용 모드에서 음성 파형을 생성한다.
with torch.inference_mode():
    output = model(**inputs).waveform
# 파형을 CPU의 1차원 NumPy 배열로 변환한다.
audio = output.squeeze().float().cpu().numpy()
# 모델 설정에서 실제 출력 샘플링 레이트를 읽는다.
sampling_rate = model.config.sampling_rate
# 파형을 WAV 파일로 저장한다.
sf.write("tts_korean.wav", audio, sampling_rate)
# 저장 결과를 출력한다.
print(f"TTS 저장 완료: tts_korean.wav / {sampling_rate} Hz")
```

**실행 참고:** 모델 체크포인트·토크나이저 로딩은 라이브러리 버전과 네트워크 환경에 영향을 받는다. 모델 최초 다운로드가 필요하며, 합성 결과의 발음 품질은 문장에 따라 달라진다. 상기 코드는 API 사용 형태를 검토한 예시이며, 이 환경에서 모델 가중치를 내려받아 실제 추론까지 실행한 것은 아니다.

## 9.1 TTS 품질 비교 과제

- 같은 문장을 문장부호만 바꾸어 합성하고 휴지·억양 차이를 기록한다.
- 숫자, 날짜, 약어, 영어 혼용 문장에 대한 발음을 평가한다.
- 문장 길이를 바꾸어 생성 시간과 실제 음성 길이를 측정한다.
- 최소 5명의 청취자가 자연스러움과 명료성을 1~5점으로 평가한다.

---

# 10. STT–LLM–TTS 음성 대화 아키텍처

```text
[사용자 음성]
      ↓
[마이크 / 브라우저 / 앱]
      ↓
[VAD / 음성 전처리]
      ↓
[STT: Whisper 등]
      ↓
[텍스트 질문]
      ↓
[LLM: GPT / Qwen / 로컬 LLM]
      ↓
[답변 텍스트]
      ↓
[TTS: VITS / XTTS 등]
      ↓
[음성 스트리밍 또는 WAV]
      ↓
[브라우저 / 스피커]
```

원문은 FastAPI를 통한 REST API 서비스 구조를 제안한다. 실제 서비스에서는 아래 사항을 추가 설계해야 한다.

1. **입력 API**: `POST /voice-chat`에 음성 파일을 업로드한다.
2. **STT 단계**: 파일 검증, 음성 구간 추출, 텍스트 전사.
3. **LLM 단계**: 시스템 지침, 대화 이력, 필요 시 RAG 문맥을 반영하여 응답 생성.
4. **TTS 단계**: 응답 문장을 음성으로 합성하고 WAV/스트림 반환.
5. **오류 처리**: 무음, 잘못된 파일, 모델 로딩 실패, 타임아웃, 장문 입력.
6. **운영 관리**: 로그·개인정보 최소화, 캐시, 지연시간, 동시 요청 제어.

## 10.1 직렬형과 스트리밍형 비교

| 구분 | 직렬형 | 스트리밍형 |
|---|---|---|
| 처리 방식 | STT 완료 → LLM 완료 → TTS 완료 | 음성·토큰·오디오를 청크 단위로 순차 전달 |
| 구현 난이도 | 낮음 | 높음 |
| 최초 응답 지연 | 상대적으로 큼 | 낮출 여지가 있음 |
| 적합한 실습 | 첫 통합 프로젝트 | 실시간 음성 비서 심화 프로젝트 |

## 10.2 FastAPI 최소 API 설계 예시

```text
POST /stt       → 음성 업로드 → JSON {"text": "..."}
POST /tts       → JSON {"text": "..."} → audio/wav
POST /voice-chat → 음성 업로드 → STT → LLM → TTS → audio/wav
GET  /health    → 서비스 상태 확인
```

이는 구현 설계 명세이며 완성된 서버 코드가 아니다. 실제 API 구현에는 업로드 파일 검증, 모델 로딩 수명주기, 비동기 처리 및 인증이 추가로 필요하다.

---

# 11. 개인정보·윤리·배포 고려사항 

- 음성에는 개인의 신원과 관련된 정보가 포함될 수 있으므로 수집·보관 목적과 동의를 명확히 한다.
- 음성 복제(Voice Cloning)는 화자의 명시적 허락을 얻은 경우에만 사용한다.
- 실제 인물의 발화를 가장하는 오용을 방지하기 위해 고지, 접근 통제, 추적 및 사용 정책을 마련한다.
- 클라우드 STT/TTS 사용 시 전송 데이터, 저장 기간, 지역 및 계약 조건을 확인한다.
- 의료·법률·금융 등 고위험 영역의 자동 전사 결과는 사람의 검토 절차를 마련한다.
- 모델의 라이선스와 데이터셋의 재사용 조건을 각각 확인한다.

---


### 추가 학습용 공식 자료 및 원 논문

- OpenAI, *Whisper* (2022): https://github.com/openai/whisper
- Radford et al., *Robust Speech Recognition via Large-Scale Weak Supervision* (2022): https://arxiv.org/abs/2212.04356
- Baevski et al., *wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations* (2020): https://arxiv.org/abs/2006.11477
- Gulati et al., *Conformer: Convolution-augmented Transformer for Speech Recognition* (2020): https://arxiv.org/abs/2005.08100
- Shen et al., *Natural TTS Synthesis by Conditioning WaveNet on Mel Spectrogram Predictions* (2017): https://arxiv.org/abs/1712.05884
- Ren et al., *FastSpeech 2: Fast and High-Quality End-to-End Text to Speech* (2020): https://arxiv.org/abs/2006.04558
- Kim et al., *Conditional Variational Autoencoder with Adversarial Learning for End-to-End Text-to-Speech* (2021): https://arxiv.org/abs/2106.06103
- Hugging Face Transformers 문서: https://huggingface.co/docs/transformers/index
- Hugging Face 한국어 MMS-TTS 모델 카드: https://huggingface.co/facebook/mms-tts-kor
- jiwer: https://github.com/jitsi/jiwer

> **검증 범위:** Markdown의 구성·표·수식·코드 문법을 점검하였다. 모델 체크포인트 다운로드 및 실제 음성 추론은 실행 환경·네트워크·모델 버전의 영향을 받으므로 별도 실행 확인이 필요하다.
