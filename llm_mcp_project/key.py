import os
import os.path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow

# Google Calendar 권한 범위 (클래스 상단 선언부와 동일하게 지정)
SCOPES = ["https://www.googleapis.com/auth/calendar"]

def create_token_file(credentials_file: str = "credentials.json", token_file: str = "token.json"):
    creds = None

    # 1. 기존에 token.json 파일이 이미 존재하는지 확인
    if os.path.exists(token_file):
        creds = Credentials.from_authorized_user_file(token_file, SCOPES)
        print(f"ℹ️ 이미 '{token_file}' 파일이 존재합니다.")

    # 2. 토큰이 없거나 유효하지 않은 경우 OAuth 인증 진행
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            print("🔄 만료된 토큰을 갱신합니다...")
            creds.refresh(Request())
        else:
            if not os.path.exists(credentials_file):
                print(f"❌ 에러: '{credentials_file}' 파일이 존재하지 않습니다.")
                print("   GCP 콘솔에서 OAuth 클라이언트 ID JSON을 다운로드하여 해당 경로에 넣어주세요.")
                return

            print("🔑 구글 계정 로그인 및 인증 승인을 시작합니다...")
            flow = InstalledAppFlow.from_client_secrets_file(credentials_file, SCOPES)
            # 웹 브라우저를 열어 사용자 로그인 유도
            creds = flow.run_local_server(port=0)

        # 3. 승인받은 자격 증명을 token.json 파일로 저장
        with open(token_file, "w") as token:
            token.write(creds.to_json())
            print(f"✅ '{token_file}' 파일이 성공적으로 만들어졌습니다!")

if __name__ == "__main__":
    create_token_file()