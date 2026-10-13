# FastAPI TTS 프로젝트

## Windows 실행

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# .env 파일에 실제 API 키 입력
python -m uvicorn main:app --reload
```

웹 화면: http://127.0.0.1:8000

API 문서: http://127.0.0.1:8000/docs

## Notebook
`FastAPI_TTS_실습.ipynb`는 같은 폴더에서 실행한다. API 호출은 유효한 OpenAI API 키 및 네트워크 연결이 필요하다.
