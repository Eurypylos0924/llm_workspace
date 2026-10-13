# 음성 기반 MNIST 숫자 생성 프로젝트

**FastAPI + OpenAI STT/TTS + PyTorch Conditional GAN**

사용자가 **"7 숫자 이미지를 생성해 줘"**라고 말하면, 음성 인식(STT) → 숫자 추출 → 조건부 GAN 이미지 생성 → TTS 안내를 수행한다.

## 1. 전체 구조

```text
브라우저 마이크 → MediaRecorder → POST /api/stt
                                  ↓
                      OpenAI 음성 인식(STT)
                                  ↓
                          한국어 명령 파싱
                                  ↓
                 POST /api/generate (digit=7)
                                  ↓
                PyTorch Conditional Generator
                                  ↓
                      PNG 이미지 웹 화면 표시
                                  ↓
                     POST /api/tts → MP3 재생
```

**중요:** 일반 GAN의 `gan_generator.pth`는 숫자 레이블로 출력을 제어할 수 없다. 이 프로젝트는 별도로 학습한 `models/cgan_generator.pth`를 사용한다. MNIST 이미지 생성 모델은 로컬에서 추론하고, STT/TTS는 OpenAI API를 사용한다. STT/TTS API 사용량에 따른 비용이 발생할 수 있다.

## 2. 파일 구조

```text
gan_voice_app/
├── app/
│   ├── main.py               # FastAPI, STT, TTS, CGAN 추론
│   ├── templates/index.html  # 웹 페이지
│   └── static/
│       ├── app.js            # 녹음, API 호출, 결과 출력
│       └── style.css         # 화면 스타일
├── training/
│   ├── __init__.py
│   ├── model.py              # 조건부 Generator/Discriminator
│   └── train_cgan.py        # MNIST CGAN 학습
├── models/                  # 학습한 체크포인트 저장
├── requirements.txt
└── README.md
```

## 3. Windows PowerShell 실행

프로젝트 폴더에서 아래 명령을 실행한다.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
Copy-Item .env.example .env
```

`.env` 파일의 `OPENAI_API_KEY`를 실제 키로 변경한다. 키는 브라우저 JavaScript나 Git 저장소에 넣지 않는다.

CUDA가 필요하다면 https://pytorch.org/get-started/locally/ 에서 환경에 맞는 PyTorch 설치 명령을 확인한다.

## 4. CGAN 모델 학습

```powershell
python -m training.train_cgan --epochs 30
```

- MNIST 60,000장을 사용한다.
- `models/cgan_generator.pth`에 학습된 Generator가 저장된다.
- `models/sample_epoch_001.png`, `sample_epoch_005.png` 등 샘플 이미지가 저장된다.
- 처음 학습할 때 데이터 다운로드가 필요하다.
- CPU 환경에서는 학습 시간이 오래 걸릴 수 있다.
- 30 Epoch는 시작값이다. 이미지 품질과 숫자 정확도가 충분하지 않다면 학습 시간을 늘리거나 모델을 개선해야 한다.

## 5. FastAPI 실행

```powershell
python -m uvicorn app.main:app --reload --host 127.0.0.1 --port 8000
```

브라우저에서 `http://127.0.0.1:8000`을 연다. API 문서는 `http://127.0.0.1:8000/docs`이다.

1. 마이크 사용을 허용한다.
2. **녹음 시작** 버튼을 누른다.
3. "숫자 7 이미지를 생성해 줘"라고 말한다.
4. **녹음 종료 및 인식** 버튼을 누른다.
5. 숫자 이미지가 생성되면 TTS 안내 음성을 재생한다.

마이크 없이도 숫자 선택창과 **선택 숫자 생성** 버튼으로 이미지를 생성할 수 있다. 다만 이 경우에도 TTS API는 호출하므로 API 키가 없으면 TTS 오류 메시지가 표시된다.

## 6. API 안내

| API | 방식 | 역할 |
|---|---|---|
| `/` | GET | 웹 화면 |
| `/api/health` | GET | 서버 및 가중치 준비 상태 |
| `/api/stt` | POST multipart | 음성 파일 → 한국어 텍스트 및 숫자 |
| `/api/generate` | POST JSON | 숫자 조건 → PNG 이미지 |
| `/api/tts` | POST JSON | 숫자 생성 완료 문장 → MP3 |

`POST /api/generate` 예시:

```json
{"digit": 7, "count": 4}
```

## 7. 주의사항

- 음성 파일은 STT 처리 과정에서 OpenAI 서비스로 전송된다.
- 마이크 녹음은 일반적으로 `localhost` 또는 HTTPS에서 사용 가능하다.
- 브라우저에서 자동 음성 재생이 차단되면 오디오 재생 버튼을 누른다.
- 녹음 형식은 브라우저가 지원하는 WebM 또는 MP4를 사용한다.
- 한국어 숫자 추출은 교육용 규칙 기반 파서이다. 복잡한 문장에서는 숫자를 잘못 인식할 수 있다.
- CGAN은 생성 결과가 요청 숫자와 항상 일치한다고 보장하지 않는다. 샘플 이미지와 숫자 분류 모델을 이용한 검증이 필요하다.
- 공개 서비스로 배포할 경우 인증, 요청 제한, 업로드 검사, API 비용 제한을 추가해야 한다.
- 모델 가중치 파일은 ZIP에 포함되지 않는다. 사용자가 직접 학습해야 한다.
