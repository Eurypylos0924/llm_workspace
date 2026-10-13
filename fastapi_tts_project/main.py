"""FastAPI + OpenAI TTS 예제. 실행: python -m uvicorn main:app --reload"""
import os
from io import BytesIO
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from openai import OpenAI, OpenAIError
from pydantic import BaseModel, Field

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')
app = FastAPI(title='FastAPI TTS', description='텍스트를 MP3 음성으로 변환')
app.mount('/static', StaticFiles(directory=BASE_DIR / 'static'), name='static')
templates = Jinja2Templates(directory=BASE_DIR / 'templates')

class TTSRequest(BaseModel):
    text: str = Field(min_length=1, max_length=4000)
    voice: Literal['alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'] = 'alloy'
    speed: float = Field(default=1.0, ge=0.25, le=4.0)

@app.get('/', response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(request=request, name='index.html')

@app.post('/api/tts')
def generate_tts(payload: TTSRequest):
    if not payload.text.strip():
        raise HTTPException(status_code=422, detail='변환할 텍스트를 입력하세요.')
    if not os.getenv('OPENAI_API_KEY'):
        raise HTTPException(status_code=500, detail='서버 API 키가 설정되지 않았습니다.')
    try:
        client = OpenAI()
        with client.audio.speech.with_streaming_response.create(
            model='tts-1', voice=payload.voice, input=payload.text,
            speed=payload.speed, response_format='mp3',
        ) as response:
            audio_bytes = response.read()
    except OpenAIError as exc:
        raise HTTPException(status_code=502, detail='음성 합성 API 요청에 실패했습니다.') from exc
    return StreamingResponse(
        BytesIO(audio_bytes), media_type='audio/mpeg',
        headers={'Content-Disposition': 'attachment; filename="tts_output.mp3"', 'Cache-Control': 'no-store'},
    )


# ============================================================
# FastAPI 서버 실행
# ============================================================
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    )