
# 파이썬 기본 경고 메시지(DeprecationWarning, UserWarning 등)가 콘솔/화면에 출력되지 않도록 숨김 처리합니다.
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# 운영체제 환경변수 접근, 정규표현식 처리, URL 인코딩 및 HTTP 요청 처리를 위한 파이썬 표준 라이브러리를 불러옵니다.
import os
import re
import urllib.parse
import urllib.request

# .env 파일에 저장된 비밀 키(예: nvidiaapi_key)를 로드하는 라이브러리를 불러옵니다.
from dotenv import load_dotenv

# LangChain에서 메모리 상에 대화 히스토리를 저장하고 관리하는 클래스를 불러옵니다.
from langchain_core.chat_history import InMemoryChatMessageHistory

# LangChain에서 프롬프트 템플릿 및 대화 기록 위치를 지정하는 모듈을 불러옵니다.
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# 대화 기록(History)을 자동으로 추적/연결해 주는 체인 실행기 객체를 불러옵니다.
from langchain_core.runnables.history import RunnableWithMessageHistory

# LangChain의 OpenAI 호환 대화형 언어 모델(LLM) 연동 클래스를 불러옵니다.
from langchain_openai import ChatOpenAI

# Streamlit 웹 대화형 UI 프레임워크를 불러옵니다.
import streamlit as st

# .env 파일에서 환경변수(nvidiaapi_key)를 읽어와 시스템 환경변수로 설정합니다.
load_dotenv()

# ---------------------------------------------------------
# 1. Streamlit UI 웹페이지 및 세션 상태 초기화
# ---------------------------------------------------------

# 브라우저 탭의 제목, 아이콘, 페이지 레이아웃(wide: 화면 전체 넓이 사용)을 기본 설정합니다.
st.set_page_config(
    page_title="마음쉼터 - 감성 챗봇 & 뮤직 플레이어",
    page_icon="🎵",
    layout="wide",  # 사이드바와 메인 대화창 활용을 위해 wide 레이아웃 적용
)

# 세션 상태(st.session_state)에 대화 기록 저장소('store') 딕셔너리가 없으면 새로 생성합니다.
if "store" not in st.session_state:
    st.session_state.store = {}

# 세션 상태에 메시지별 추천 음악 데이터 저장소('music_data') 딕셔너리가 없으면 생성합니다.
if "music_data" not in st.session_state:
    st.session_state.music_data = {}

# 세션 상태에 현재 사이드바에서 재생 중인 음악 URL('current_playing_url') 변수가 없으면 None으로 초기화합니다.
if "current_playing_url" not in st.session_state:
    st.session_state.current_playing_url = None

# 세션 상태에 현재 사이드바에서 재생 중인 음악 제목('current_playing_title') 변수가 없으면 None으로 초기화합니다.
if "current_playing_title" not in st.session_state:
    st.session_state.current_playing_title = None


# 고유 세션 ID에 해당하는 대화 히스토리 객체를 반환하거나, 없으면 신규 생성하는 관리 함수입니다.
def get_session_history(session_id: str):
    # 전달받은 session_id가 저장소에 존재하지 않는 경우
    if session_id not in st.session_state.store:
        # 인메모리 대화 히스토리 객체를 생성합니다.
        history = InMemoryChatMessageHistory()
        # 첫 방문 사용자를 위한 AI 심리상담사의 기본 첫 인사 메시지를 추가합니다.
        history.add_ai_message(
            "안녕하세요. 오늘 하루는 어떠셨나요? ☕\n편안하게 당신의 마음을"
            " 나누어 주세요."
        )
        # 생성된 히스토리 객체를 세션 저장소 딕셔너리에 저장합니다.
        st.session_state.store[session_id] = history
    # 해당 세션 ID의 대화 히스토리 객체를 반환합니다.
    return st.session_state.store[session_id]


# ---------------------------------------------------------
# 2. 고속 YouTube 동영상 검색 함수
# ---------------------------------------------------------


# 입력된 검색어(query)로 유튜브 결과를 크롤링하여 동영상 URL 및 제목 정보를 수집하는 함수입니다.
def search_youtube_videos(query: str, max_results: int = 5) -> list[dict]:
    try:
        # 검색어 뒤에 " 음악"을 붙인 후 웹 URL 규칙에 맞게 퍼센트 인코딩(URL Encoding)합니다.
        search_query = urllib.parse.quote(query + " 음악")
        # 유튜브 검색 결과 페이지 요청 URL 문자열을 생성합니다.
        url = f"https://www.youtube.com/results?search_query={search_query}"

        # 유튜브 서버의 봇 차단을 방지하기 위해 일반 크롬 브라우저 헤더(User-Agent) 정보를 포함한 Request 객체를 생성합니다.
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
                    " AppleWebKit/537.36 (KHTML, like Gecko)"
                    " Chrome/120.0.0.0 Safari/537.36"
                )
            },
        )

        # HTTP 요청을 보내고 응답 결과 HTML 페이지 전체 소스 코드를 UTF-8 텍스트로 변환하여 저장합니다.
        html = urllib.request.urlopen(req).read().decode("utf-8")
        # 정규표현식을 사용해 HTML 소스 코드 내에서 11자리 고유 비디오 ID 패턴('watch?v=비디오ID')을 모두 찾아 리스트로 반환합니다.
        video_ids = re.findall(r"watch\?v=([a-zA-Z0-9_-]{11})", html)

        # 파싱 결과를 담을 리스트와 중복 ID 검증용 세트(Set)를 선언합니다.
        results = []
        seen_ids = set()

        # 추출된 비디오 ID 목록을 순회 처리합니다.
        for vid in video_ids:
            # 이미 처리한 중복 비디오 ID가 아닌 경우에만 실행합니다.
            if vid not in seen_ids:
                # 중복 방지를 위해 집합(set)에 해당 비디오 ID를 등록합니다.
                seen_ids.add(vid)
                # 제목과 해당 유튜브 영상의 full URL 주소를 딕셔너리 형태로 리스트에 추가합니다.
                results.append({
                    "title": f"유튜브 음악 트랙 ({vid})",
                    "url": f"https://www.youtube.com/watch?v={vid}",
                })
                # 원하는 최대 수집 개수(max_results)에 도달하면 반복문을 탈출합니다.
                if len(results) >= max_results:
                    break

        # 완성된 수집 결과 딕셔너리 리스트를 반환합니다.
        return results
    except Exception:
        # 네트워크 오류 등 예외 발생 시 빈 리스트를 반환합니다.
        return []


# ---------------------------------------------------------
# 3. 사이드바 고정 음악 플레이어 영역
# ---------------------------------------------------------

# Streamlit 사이드바(왼쪽 고정 화면) UI 구역을 정의합니다.
with st.sidebar:
    # 사이드바 상단 헤더 타이틀을 표시합니다.
    st.header("🎧 마음쉼터 뮤직 오디오")
    # 플레이어에 대한 부연 안내 캡션 문구를 표시합니다.
    st.caption("대화 중에도 음악이 끊기지 않고 계속 재생됩니다.")

    # 세션 상태에 재생 등록된 음악 URL이 존재하는 경우 사이드바 오디오 플레이어를 화면에 출력합니다.
    if st.session_state.current_playing_url:
        # 재생 중인 트랙 제목을 초록색 강조 성공 박스(success callout)로 표시합니다.
        st.success(
            f"🎵 **재생 중:** {st.session_state.current_playing_title}"
        )
        # Streamlit 비디오/오디오 플레이어 컴포넌트를 통해 해당 유튜브 URL 영상을 재생합니다.
        st.video(st.session_state.current_playing_url)
    else:
        # 아직 선택된 음악이 없을 경우 안내 파란색 정보 박스(info callout)를 표시합니다.
        st.info("💡 대화창에서 위로의 추천 음악을 선택해 보세요!")

    # 구분선 스타일을 적용합니다.
    st.markdown("---")


# ---------------------------------------------------------
# 4. 음악 선택 목록 렌더링 함수
# ---------------------------------------------------------


# 메인 대화창에 추천된 음악 목록을 라디오 버튼 옵션으로 그려주는 렌더링 함수입니다.
def render_music_selector(music_list: list, msg_idx: int):
    # visual 구분선을 추가합니다.
    st.markdown("---")
    # 추천 음악 섹션의 서브헤더 제목을 표시합니다.
    st.subheader("🎵 추천 음악 리스트")

    # 라디오 버튼에 보여줄 '선택 라벨 문자열': '음악 딕셔너리 정보' 구조의 맵핑 딕셔너리를 작성합니다.
    options_dict = {
        f"트랙 #{i+1} : {item['url']}": item
        for i, item in enumerate(music_list)
    }

    # Streamlit 위젯 상태 충돌을 방지하기 위해 메시지 인덱스 기반의 고유한 key 이름을 지정합니다.
    radio_key = f"radio_{msg_idx}"

    # 화면에 트랙 선택 라디오 버튼 단일 선택 위젯을 생성하고 선택된 라벨 옵션을 받습니다.
    selected_label = st.radio(
        "듣고 싶은 트랙을 클릭하면 오른쪽 사이드바에서 연속 재생됩니다:",
        options=list(options_dict.keys()),
        key=radio_key,
    )

    # 사용자가 라디오 버튼 항목을 선택했을 때 실행되는 조건문입니다.
    if selected_label:
        # 선택한 라벨 문자열에 맵핑된 음악 정보 딕셔너리를 추출합니다.
        selected_item = options_dict[selected_label]
        # 현재 사이드바에서 재생 중인 URL과 새로 선택한 URL이 다른 경우에만 실행합니다.
        if (
            st.session_state.current_playing_url != selected_item["url"]
        ):
            # 세션 상태의 재생 URL을 새 음악 URL로 교체합니다.
            st.session_state.current_playing_url = selected_item["url"]
            # 세션 상태의 재생 제목을 새 트랙 라벨 문자열로 교체합니다.
            st.session_state.current_playing_title = selected_label
            # 사이드바 플레이어에 바뀐 음악을 즉시 반영하여 재생시키기 위해 페이지 전체를 즉시 재실행(Rerun)합니다.
            st.rerun()


# ---------------------------------------------------------
# 5. LangChain 감성 상담 챗봇 구성
# ---------------------------------------------------------

# AI 감성 상담사의 페르소나 및 응답 지침, 대화 맥락 플레이스홀더를 담은 프롬프트 템플릿을 정의합니다.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """당신은 공감 능력이 뛰어난 따뜻한 감성 상담사입니다.
사용자의 이야기에서 기분이나 상태를 파악해 따뜻하게 위로와 공감을 전하세요.
답변 마지막 줄에는 반드시 사용자의 기분에 맞는 음악 검색 키워드를 아래 형식으로 작성해 주세요.

검색키워드: [기분/상태 관련 음악 검색어]
(예: 검색키워드: 그리울 때 듣는 잔잔한 노래)""",
        ),
        # 대화 히스토리 목록이 주입될 위치 플레이스홀더 지정
        MessagesPlaceholder(variable_name="history"),
        # 사용자 신규 입력을 받기 위한 인간(human) 메시지 지정
        ("human", "{input}"),
    ]
)

# NVIDIA(OpenAI 호환 API)의 gpt-oss-20b 모델을 인스턴스화합니다. (temperature=0.7로 창의적이고 따뜻한 응답 유도)
llm = ChatOpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.getenv("nvidiaapi_key"),
    model="openai/gpt-oss-20b",
    temperature=0.7,
)
# 프롬프트 템플릿과 LLM 모델을 파이프라인(|) 기호로 결합하여 기본 체인을 작성합니다.
chain = prompt | llm

# 세션 히스토리 자동 관리 기능이 결합된 최종 Runnable 대화 체인 객체를 생성합니다.
bot_with_history = RunnableWithMessageHistory(
    chain,
    get_session_history,
    input_messages_key="input",
    history_messages_key="history",
)

# ---------------------------------------------------------
# 6. 메인 화면 대화 기록 및 UI 구성
# ---------------------------------------------------------

# 메인 화면에 대화형 애플리케이션의 타이틀 제목을 표시합니다.
st.title("🎵 마음쉼터: 감성 챗봇 & 음악 플레이어")
# 메인 서브 안내 문구를 캡션 형태로 표시합니다.
st.caption(
    "오늘 어떤 기분이신가요? 마음을 이야기해 주시면 따뜻한 위로와 음악을"
    " 추천해 드립니다."
)

# 세션 상태에서 감성 상담 대화용 'emotional_music_session' 히스토리 객체를 가져옵니다.
session_history = get_session_history("emotional_music_session")

# 저장된 대화 기록(messages)을 순회하면서 이전 대화 내역들을 순서대로 화면에 출력합니다.
for idx, msg in enumerate(session_history.messages):
    # 메시지 타입이 'ai'이면 assistant, 아니면 user 역할을 지정합니다.
    role = "assistant" if msg.type == "ai" else "user"

    # 챗봇 메시지 스타일 컨테이너(chat_message)를 생성합니다.
    with st.chat_message(role):
        # AI 메시지의 경우 프롬프트 지침용 '검색키워드:' 문구를 분리하여 위로 텍스트 본문만 화면에 출력합니다.
        content_to_show = msg.content.split("검색키워드:")[0].strip()
        st.write(content_to_show)

        # AI 상담사의 메시지이면서 세션 음악 데이터에 해당 메시지 인덱스의 음악이 존재하는 경우
        if role == "assistant" and idx in st.session_state.music_data:
            # 세션에 저장되어 있던 추천 음악 트랙 목록 데이터를 가져옵니다.
            music_list = st.session_state.music_data[idx]
            # 음악 데이터 리스트가 비어있지 않은 경우 라디오 선택 플레이어 UI를 출력합니다.
            if music_list:
                render_music_selector(music_list, idx)

# ---------------------------------------------------------
# 7. 사용자 신규 입력 처리
# ---------------------------------------------------------

# 화면 하단 채팅 입력창(st.chat_input)에서 사용자의 입력값을 받으면 조건문 이하를 실행합니다.
if user_input := st.chat_input("당신의 이야기를 들려주세요..."):
    # 사용자가 입력한 메시지를 대화창 화면에 'user' 역할 스타일로 즉시 출력합니다.
    st.chat_message("user").write(user_input)

    # LangChain 대화 히스토리 구분을 위한 세션 식별 ID 세팅값을 딕셔너리로 준비합니다.
    config = {"configurable": {"session_id": "emotional_music_session"}}

    # 'assistant' 챗봇 역할 스타일 컨테이너 내에서 AI 응답 생성을 수행합니다.
    with st.chat_message("assistant"):
        # LangChain 대화 체인을 실행하여 사용자 입력에 대한 답변 객체를 생성 받아옵니다.
        response = bot_with_history.invoke({"input": user_input}, config=config)
        # LLM 응답 텍스트 전체 원문을 문자열 변수에 저장합니다.
        full_response = response.content

        search_keyword = ""
        display_text = full_response

        # AI 응답 내에 '검색키워드:' 구분 문구가 포함되어 포함된 경우 실행합니다.
        if "검색키워드:" in full_response:
            # '검색키워드:' 텍스트 기준으로 분할합니다.
            parts = full_response.split("검색키워드:")
            # 첫 번째 파트(상담 위로 문구)를 사용자 표시용 텍스트로 저장합니다.
            display_text = parts[0].strip()
            # 두 번째 파트(키워드)를 유튜브 음악 검색용 키워드로 지정합니다.
            search_keyword = parts[1].strip()
        else:
            # 구분 키워드가 없는 예외 상황의 경우 사용자 입력 문장에 "음악"을 붙여 기본 검색 키워드로 사용합니다.
            search_keyword = f"{user_input} 음악"

        # 파싱 완료된 AI 상담사의 위로 문구를 화면에 렌더링 출력합니다.
        st.write(display_text)

        # 현재 세션 메시지 히스토리에 방금 새로 추가된 AI 메시지의 인덱스 번호를 계산합니다.
        new_ai_index = len(session_history.messages) - 1

        # 유튜브 음악 검색 진행 동안 동작할 로딩 스피너(st.spinner)를 보여줍니다.
        with st.spinner(
            f"🎶 '{search_keyword}' 관련 음악을 검색 중입니다..."
        ):
            # 유튜브 동영상 실시간 고속 검색 함수를 호출하여 5개의 추천 음악 목록을 가져옵니다.
            music_list = search_youtube_videos(search_keyword)

        # 검색된 추천 음악 리스트를 세션의 메시지 인덱스 키값으로 보존 저장합니다.
        st.session_state.music_data[new_ai_index] = music_list

        # 수집된 검색 결과 음악 목록이 존재하는 경우 자동으로 첫 번째 트랙을 재생 등록합니다.
        if music_list:
            # 첫 번째 트랙의 URL을 세션 상태의 현재 재생 URL로 설정합니다.
            st.session_state.current_playing_url = music_list[0]["url"]
            # 첫 번째 트랙의 라벨 정보를 세션 상태의 현재 재생 제목으로 설정합니다.
            st.session_state.current_playing_title = (
                f"트랙 #1 : {music_list[0]['url']}"
            )
            # 메인 대화창 하단에 음악 선택 라디오 위젯 UI를 출력합니다.
            render_music_selector(music_list, new_ai_index)
            # 새로 등록된 음악을 사이드바 플레이어에 바로 적용 및 자동 재생하기 위해 앱을 재실행합니다.
            st.rerun()
        else:
            # 검색 결과가 존재하지 않을 때 안내 메시지를 출력합니다.
            st.info("관련 음악 영상 검색 결과를 불러오지 못했습니다.")
