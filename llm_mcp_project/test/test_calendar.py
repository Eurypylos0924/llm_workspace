# test_calendar.py
from external_adapter.google_calendar import GoogleCalendarAdapter

if __name__ == "__main__":
    print("Google Calendar 인증 시작...")
    calendar = GoogleCalendarAdapter()
    
    # 1. 일정 조회 테스트 (최초 실행 시 여기서 브라우저가 열립니다)
    events = calendar.list_events(max_results=5)
    print("\n[조회된 향후 일정]")
    print(events)