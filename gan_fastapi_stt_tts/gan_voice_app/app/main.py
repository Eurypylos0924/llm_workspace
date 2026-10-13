"""FastAPI + OpenAI STT/TTS + MNIST Conditional GAN 웹 서버."""
import io
import os
import re
from pathlib import Path
from threading import Lock

import torch
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from openai import AsyncOpenAI, OpenAIError
from PIL import Image
from pydantic import BaseModel, Field

from training.model import ConditionalGenerator, LATENT_DIM


load_dotenv()
# 프로젝트 루트를 기준으로 경로를 계산하여 어느 폴더에서 실행하든 동일하게 동작시킨다.
ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = ROOT / "models" / "cgan_generator.pth"
# API 키는 소스코드에 직접 입력하지 않고 환경변수로 읽는다.
API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
STT_MODEL = os.getenv("STT_MODEL", "gpt-4o-mini-transcribe")
TTS_MODEL = os.getenv("TTS_MODEL", "tts-1")
TTS_VOICE = os.getenv("TTS_VOICE", "alloy")
# 파일 업로드 크기를 제한하여 과도한 메모리 사용을 방지한다.
MAX_AUDIO_BYTES = 15 * 1024 * 1024
# 단일 서버 프로세스에서 Generator를 재사용하도록 모델 캐시를 유지한다.
_model = None
_model_lock = Lock()

app = FastAPI(title="Voice MNIST CGAN", description="음성으로 숫자를 지정하여 CGAN 이미지 생성")
app.mount("/static", StaticFiles(directory=ROOT / "app" / "static"), name="static")


class GenerateRequest(BaseModel):
    # 생성할 숫자는 반드시 0~9 범위의 정수여야 한다.
    digit: int = Field(ge=0, le=9)
    # 서로 다른 노이즈를 사용해 생성할 이미지 개수를 지정한다.
    count: int = Field(default=1, ge=1, le=16)


def get_device() -> torch.device:
    # 서버 추론 장치를 CUDA, MPS, CPU 순서로 결정한다.
    return torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")


def get_generator() -> tuple[ConditionalGenerator, torch.device]:
    global _model
    # 여러 요청이 동시에 들어와도 모델을 한 번만 초기화하도록 잠금을 사용한다.
    with _model_lock:
        if _model is not None:
            return _model
        if not MODEL_PATH.is_file():
            raise HTTPException(status_code=503, detail="CGAN 가중치가 없습니다. 먼저 python -m training.train_cgan --epochs 30 을 실행하세요.")
        device = get_device()
        # 직접 학습한 체크포인트만 읽는다. 외부에서 받은 임의의 .pth 파일은 로딩하지 않는다.
        try:
            checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=True)
            # ConditionalGenerator() : Conditional GAN(cGAN, 조건부 생성적 적대 신경망)의 Generator 객체를 생성한다.
            # 입력 : 노이즈 z + 레이블 y
            # 출력 : 지정한 숫자 이미지
            model = ConditionalGenerator().to(device)

            # checkpoint : 학습 결과를 저장
            # generator_state_dict : Generator의 가중치와 편향등이 저장된 항목
            # strict : 저장된 파라미터 이름이 현재 모델과 정확하게 일치해야 함
            model.load_state_dict(checkpoint["generator_state_dict"], strict=True)
            # 평가를 하기 위해서 model.eval()로 설정
            model.eval()
        except (RuntimeError, KeyError, ValueError, TypeError, OSError) as exc:
            raise HTTPException(status_code=503, detail=f"모델 가중치 로딩 실패: {exc}") from exc
        
        _model = (model, device)
        return _model


def parse_digit(text: str) -> int:
    # '7', '숫자 7', '7숫자' 등 숫자로 입력된 명령을 먼저 검색한다.
    # (?<!\d) : 앞에 숫자가 없는 경우 왼쪽
    # [0-9] : 숫자 0~9
    # (?!\d) : 뒤에 숫자가 없는 경우 오른쪽 

    matches = re.findall(r"(?<!\d)[0-9](?!\d)", text)
    # '칠', '일곱', '영', '공' 등 한국어 숫자 발화도 처리한다.
    korean_numbers = {"영": 0, "공": 0, "제로": 0, "일": 1, "하나": 1, "이": 2, "둘": 2, "삼": 3, "셋": 3, "사": 4, "넷": 4, "오": 5, "다섯": 5, "육": 6, "여섯": 6, "륙": 6, "칠": 7, "일곱": 7, "팔": 8, "여덟": 8, "구": 9, "아홉": 9}
    # '일'과 '이'처럼 일상 단어에 포함되기 쉬운 한 글자 표현은 숫자 문맥에서만 인식한다.
    contextual = bool(re.search(r"숫자|번호|이미지|그림|생성|만들|그려", text))
    word_pattern = "|".join(sorted(map(re.escape, korean_numbers), key=len, reverse=True))
    if contextual:
        # '일곱 숫자' 및 '숫자 칠' 같은 명령에서 숫자 단어를 인식한다.
        matches += [str(korean_numbers[m.group()]) for m in re.finditer(rf"(?<![가-힣])(?:{word_pattern})(?![가-힣])", text)]
    unique = set(matches)
    if len(unique) == 1:
        return int(next(iter(unique)))
    if not unique:
        raise HTTPException(status_code=422, detail="숫자를 인식하지 못했습니다. '숫자 7 이미지를 생성해 줘'라고 말해 주세요.")
    raise HTTPException(status_code=422, detail="여러 숫자가 감지되었습니다. 생성할 숫자 하나만 말해 주세요.")


def generate_png(digit: int, count: int) -> bytes:
    # 추론 모델과 장치를 준비한다.
    generator, device = get_generator()
    # 각 이미지에 다른 무작위 노이즈를 입력한다.
    noise = torch.randn(count, LATENT_DIM, device=device)
    labels = torch.full((count,), digit, dtype=torch.long, device=device)
    # 추론에서는 기울기 계산을 생략한다.
    with torch.inference_mode():
        images = generator(noise, labels).reshape(count, 1, 28, 28)
        images = ((images + 1) / 2 * 255).clamp(0, 255).to(torch.uint8).cpu()
    # 생성 이미지가 여러 장이면 가로로 나란히 붙여 PNG 한 장으로 제공한다.
    canvas = Image.new("L", (count * 28, 28), color=0)
    for index, image in enumerate(images):
        canvas.paste(Image.fromarray(image.squeeze(0).numpy(), mode="L"), (index * 28, 0))
    # 브라우저에서 확인하기 쉽도록 최근접 보간으로 8배 확대한다.
    canvas = canvas.resize((count * 28 * 8, 28 * 8), Image.Resampling.NEAREST)
    buffer = io.BytesIO()
    canvas.save(buffer, format="PNG")
    return buffer.getvalue()


@app.get("/")
async def index():
    # 정적 HTML 파일을 메인 화면으로 제공한다.
    return FileResponse(ROOT / "app" / "templates" / "index.html")


@app.get("/api/health")
async def health():
    # 서버 상태와 모델 가중치 준비 여부를 확인한다.
    return {"status": "ok", "model_ready": MODEL_PATH.is_file(), "stt_tts_ready": bool(API_KEY)}


@app.post("/api/stt")
async def transcribe(audio: UploadFile = File(...)):
    # OpenAI API 키가 없으면 음성 인식을 실행할 수 없다.
    if not API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY를 .env에 설정하세요.")
    # 브라우저 녹음 파일의 형식과 크기를 확인한다.
    if audio.content_type and not any(part in audio.content_type for part in ("audio/", "video/webm", "application/octet-stream")):
        raise HTTPException(status_code=415, detail="지원하지 않는 오디오 형식입니다.")
    content = await audio.read(MAX_AUDIO_BYTES + 1)
    if not content or len(content) > MAX_AUDIO_BYTES:
        raise HTTPException(status_code=413, detail="음성 파일이 비어 있거나 15MB를 초과했습니다.")
    # 브라우저의 MediaRecorder가 전송하는 webm 파일을 OpenAI에 전달한다.
    filename = Path(audio.filename or "speech.webm").name
    try:
        client = AsyncOpenAI(api_key=API_KEY)
        result = await client.audio.transcriptions.create(
            model=STT_MODEL,
            file=(filename, content, audio.content_type or "audio/webm"),
            language="ko",
        )
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"STT 서비스 오류: {exc}") from exc
    transcript = result.text.strip()
    # STT 결과를 숫자 명령으로 해석한다.
    digit = parse_digit(transcript)
    return {"transcript": transcript, "digit": digit}


@app.post("/api/generate")
async def generate(request: GenerateRequest):
    # 숫자 레이블을 조건으로 지정한 개수의 이미지를 생성한다.
    image_bytes = generate_png(request.digit, request.count)
    return Response(content=image_bytes, media_type="image/png", headers={"Cache-Control": "no-store"})


@app.post("/api/tts")
async def synthesize(request: GenerateRequest):
    # 생성 완료 메시지를 한국어 음성으로 합성한다.
    if not API_KEY:
        raise HTTPException(status_code=503, detail="OPENAI_API_KEY를 .env에 설정하세요.")
    message = f"숫자 {request.digit} 이미지 {request.count}장을 생성했습니다."
    try:
        client = AsyncOpenAI(api_key=API_KEY)
        # OpenAI SDK의 음성 생성 API를 호출하고 MP3 바이트를 읽는다.
        response = await client.audio.speech.create(model=TTS_MODEL, voice=TTS_VOICE, input=message, response_format="mp3")
        audio_bytes = response.content
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail=f"TTS 서비스 오류: {exc}") from exc
    return Response(content=audio_bytes, media_type="audio/mpeg", headers={"Cache-Control": "no-store"})
