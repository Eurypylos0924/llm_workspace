# 파일/경로 존재 여부 확인을 위해 os.path 모듈을 임포트합니다.
import os.path
# 일정 시각 계산 및 ISO 8601 변환을 위한 datetime 모듈을 임포트합니다.
from datetime import datetime
# 타입 힌팅 지원을 위한 Any, Dict, List, Optional 모듈을 임포트합니다.
from typing import Any, Dict, List, Optional
# 경로 다루기를 위한 pathlib의 Path 클래스를 임포트합니다.
from pathlib import Path

# Google OAuth 2.0 인증 요청 전달 객체를 임포트합니다.
from google.auth.transport.requests import Request
# 파일에서 사용자 자격 증명(Token)을 로드하기 위한 Credentials 클래스를 임포트합니다.
from google.oauth2.credentials import Credentials
# 웹 브라우저 기반의 로컬 서버 방식 인증 흐름(Flow)을 처리하는 객체를 임포트합니다.
from google_auth_oauthlib.flow import InstalledAppFlow
# 구글 서비스(Calendar API 등)와 통신하기 위한 클라이언트 빌더 함수를 임포트합니다.
from googleapiclient.discovery import build

# 구글 캘린더 접근 권한 범위를 설정합니다. (읽기/쓰기 통합 권한 범위)
SCOPES = ["https://www.googleapis.com/auth/calendar"]


class GoogleCalendarAdapter:
    """Google Calendar API 연동 어댑터 클래스입니다."""

    def __init__(
        self,
        credentials_file: str = "credentials.json",
        token_file: str = "token.json",
    ):
        # OAuth 2.0 앱 클라이언트 보안 비밀번호 파일 경로 저장
        self.credentials_file = credentials_file
        # 생성된 인증 토큰(Access/Refresh Token)을 저장할 파일 경로 저장
        self.token_file = token_file
        # Calendar API 서비스 객체 초기화 (lazy loading 방식으로 구현)
        self.service = None

    def _get_service(self):
        """OAuth 2.0 인증을 거쳐 Google Calendar API 서비스 객체를 반환합니다."""
        # 이미 생성된 서비스 객체가 있는 경우 재사용하여 인스턴스를 반환
        if self.service:
            return self.service

        # 자격 증명 변수 초기화
        creds = None
        
        # 이전 실행 시 생성된 토큰 파일이 존재하는지 확인 및 기존 토큰 로드
        if os.path.exists(self.token_file):
            creds = Credentials.from_authorized_user_file(self.token_file, SCOPES)

        # 토큰이 없거나 유효하지 않은 경우 새롭게 로그인/갱신 절차 수행
        if not creds or not creds.valid:
            # 토큰이 만료되었으나 Refresh Token이 존재하는 경우 토큰 자동 갱신
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                # OAuth 인증용 비밀키 파일이 존재하는지 확인 (없으면 FileNotFoundError 예외 발생)
                if not os.path.exists(self.credentials_file):
                    raise FileNotFoundError(
                        f"'{self.credentials_file}' 파일이 존재하지 않습니다. Google Cloud Console에서 다운로드 후 배치해 주세요."
                    )
                # 클라이언트 비밀키 파일 기반으로 InstalledAppFlow 인증 객체 생성
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, SCOPES
                )
                # 로컬 HTTP 서버를 띄워 사용자의 웹 브라우저 로그인 인증 진행
                creds = flow.run_local_server(port=0)

            # 새롭게 발급받은 인증 토큰을 다음 실행 시 재사용하기 위해 파일로 저장
            with open(self.token_file, "w", encoding="utf-8") as token:
                token.write(creds.to_json())

        # Google Calendar API 서비스(v3 버전) 객체를 구축하여 인스턴스 변수에 저장
        self.service = build("calendar", "v3", credentials=creds)
        return self.service

    def list_events(self, max_results: int = 10) -> List[Dict[str, Any]]:
        """가장 가까운 향후 일정 목록을 조회합니다."""
        # 구글 캘린더 API 서비스 객체를 획득합니다.
        service = self._get_service()
        # 현재 UTC 시각을 ISO 8601 포맷 문자열로 생성합니다 ('Z'는 UTC 지정자).
        now = datetime.utcnow().isoformat() + "Z"

        # 사용자의 기본(primary) 캘린더에서 현재 시각 이후의 일정을 최대 max_results 개까지 조회합니다.
        events_result = (
            service.events()
            .list(
                calendarId="primary",
                timeMin=now,
                maxResults=max_results,
                singleEvents=True,
                orderBy="startTime",
            )
            .execute()
        )
        # 결과 객체에서 항목(items) 리스트를 추출합니다.
        events = events_result.get("items", [])

        # 정제된 일정 데이터를 담을 리스트 초기화
        formatted_events = []
        for event in events:
            # 시작 시각 추출 (시간 지정 일정은 dateTime, 종일 일정은 date 필드 참조)
            start = event["start"].get("dateTime", event["start"].get("date"))
            # 종료 시각 추출 (시간 지정 일정은 dateTime, 종일 일정은 date 필드 참조)
            end = event["end"].get("dateTime", event["end"].get("date"))
            
            # API 응답 객체 중 주요 정보만 추출하여 딕셔너리로 구성 후 추가
            formatted_events.append(
                {
                    "id": event.get("id"),
                    "title": event.get("summary", "(제목 없음)"),
                    "start": start,
                    "end": end,
                    "description": event.get("description", ""),
                    "location": event.get("location", ""),
                }
            )
        return formatted_events

    def create_event(
        self,
        summary: str,
        start_time: str,
        end_time: str,
        description: str = "",
        location: str = "",
        timezone: str = "Asia/Seoul",
    ) -> Dict[str, Any]:
        """Google Calendar에 새 일정을 등록합니다.

        :param summary: 일정 제목
        :param start_time: 시작 시간 (ISO 8601 형식, 예: '2026-10-01T10:00:00')
        :param end_time: 종료 시간 (ISO 8601 형식, 예: '2026-10-01T11:00:00')
        :param description: 상세 설명
        :param location: 장소
        :param timezone: 타임존
        """
        # 구글 캘린더 API 서비스 객체를 획득합니다.
        service = self._get_service()

        # 구글 캘린더 API 규격에 맞는 일정 등록 데이터 요청 본문(body)을 구성합니다.
        event_body = {
            "summary": summary,
            "location": location,
            "description": description,
            "start": {
                "dateTime": start_time,
                "timeZone": timezone,
            },
            "end": {
                "dateTime": end_time,
                "timeZone": timezone,
            },
        }

        # 기본 캘린더(primary)에 새로운 일정을 삽입하는 API 호출 실행
        created_event = (
            service.events()
            .insert(calendarId="primary", body=event_body)
            .execute()
        )

        # 생성된 일정의 주요 결과 정보(ID, 제목, 이동링크, 시각)를 딕셔너리로 구성하여 반환
        return {
            "id": created_event.get("id"),
            "title": created_event.get("summary"),
            "htmlLink": created_event.get("htmlLink"),
            "start": start_time,
            "end": end_time,
        }