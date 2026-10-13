from fastapi import FastAPI, Request
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from openai import OpenAI
from yt_lib import trim_audio_file
from dotenv import load_dotenv
import json
import os
import asyncio
import subprocess

# API 키 불러오기
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("환경 변수 OPENAI_API_KEY가 설정되지 않았습니다.")

client = OpenAI() # 최근 API는 자동으로 환경변수를 불러와서 client를 초기화 함

app = FastAPI()

# 템플릿 디렉토리 설정
templates = Jinja2Templates(directory="templates")

# static 폴더에 있는 파일 서빙 설정
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse(name="index3.html", request= request)


@app.get("/youtube/summary/{video_id}")
async def chat_stream(video_id: str = 'gqXEEydkbh4'):
    """YouTube 오디오 다운로드 → Whisper 변환 → GPT 요약 (스트리밍 응답)"""

    async def event_stream():
        # Step 1: 유튜브 오디오 다운로드 중
        yield f"data: {json.dumps({'status': 'downloading_audio', 'message': '유튜브 오디오를 내려받고 있습니다.'}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(1) # 너무 빠르게 메시지가 전송되는 것을 방지하고 자연스럽게 출력되도록 조절.

        # YouTube 다운로드 실행 (스트리밍 로그 전송)
        process = subprocess.Popen(
            ["yt-dlp", "-x", "--audio-format", "mp3", "-o", "downloads/%(id)s.%(ext)s", f"https://www.youtube.com/watch?v={video_id}"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # yt-dlp의 로그를 실시간으로 웹 브라우저에 출력
        while True:
            output = process.stdout.readline()
            if output == "" and process.poll() is not None:
                break
            if output:
                yield f"data: {json.dumps({'status': 'downloading_audio', 'message': output.strip()}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(1) # 너무 빠르게 메시지가 전송되는 것을 방지하고 자연스럽게 출력되도록 조절.

        if process.returncode != 0:
            error_message = process.stderr.read()
            yield f"data: {json.dumps({'status': 'error', 'message': f'유튜브 다운로드 오류: {error_message}'}, ensure_ascii=False)}\n\n"
            return

        audio_path = f"downloads/{video_id}.mp3"

        # 오디오 자르기
        trimmed_audio_path = trim_audio_file(audio_path)
        yield f"data: {json.dumps({'status': 'trim_audio', 'message': '오디오 자르기가 완료되었습니다.'}, ensure_ascii=False)}\n\n"
        await asyncio.sleep(1) # 자연스로운 지연을 위해서 추가함

        # Step 2: 음성 → 텍스트 변환 중
        yield f"data: {json.dumps({'status': 'transcribing_audio', 'message': '음성을 텍스트로 변환 중입니다.'}, ensure_ascii=False)}\n\n"

        # Whisper를 이용한 음성 인식
        with open(trimmed_audio_path, "rb") as audio_file:
            text = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                response_format="text"
            )
        yield f"data: {json.dumps({'status': 'transcription_complete', 'message': '음성 변환을 완료했습니다.'}, ensure_ascii=False)}\n\n"

        # Step 3: GPT 요약 시작
        yield f"data: {json.dumps({'status': 'summarizing_text', 'message': '내용을 요약하고 있습니다.'}, ensure_ascii=False)}\n\n"

        # GPT-4o-mini 요약 (스트리밍)
        stream = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0,
            messages=[
                {"role": "system", "content": "주어진 텍스트를 간결하고 명확하게 요약해 주세요."},
                {"role": "user", "content": text}
            ],
            stream=True
        )

        for chunk in stream:
            content = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
            if content:
                yield f"data: {json.dumps({'status': 'processing', 'token': content}, ensure_ascii=False)}\n\n"
                await asyncio.sleep(0)  # 다른 작업을 수행할 수 있도록 컨텍스트 스위칭

        # Step 4: 모든 과정 완료
        yield f"data: {json.dumps({'status': 'complete', 'message': '요약이 완료되었습니다!'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    import asyncio

    async def main():
        config = uvicorn.Config(app, host="127.0.0.1", port=8000)
        server = uvicorn.Server(config)
        await server.serve()

    asyncio.run(main())
