## Windows 환경에서 FFmpeg 설치 방법

현재 `yt_dlp`을 이용해 YouTube 오디오를 다운로드하고 MP3로 변환하려면 FFmpeg 실행 파일을 설치하고 Windows 환경 변수 PATH에 등록해야 한다.

### 1. FFmpeg 다운로드

다음 사이트에서 Windows용 FFmpeg을 다운로드한다.

- 공식 사이트: [https://ffmpeg.org/download.html](https://ffmpeg.org/download.html)
- Windows 빌드: [https://github.com/BtbN/FFmpeg-Builds/releases](https://github.com/BtbN/FFmpeg-Builds/releases)

GitHub에서 다음 파일을 선택할 수 있다.

```
ffmpeg-master-latest-win64-gpl.zip
```

### 2. 압축 해제

다운로드한 ZIP 파일의 압축을 해제한 후 다음과 같이 폴더를 배치한다.

```
C:\
 └── ffmpeg\
      ├── bin\
      │    ├── ffmpeg.exe
      │    ├── ffprobe.exe
      │    └── ffplay.exe
      ├── doc\
      └── presets\
```

배포 파일에 따라 폴더 구성은 다를 수 있다. 핵심은 `ffmpeg.exe`와 `ffprobe.exe`가 들어 있는 `bin` 폴더이다.

### 3. Windows 환경 변수 PATH 설정

1. Windows 검색에서 환경 변수 편집을 검색한다.
2. 시스템 환경 변수 편집을 실행한다.
3. 환경 변수 버튼을 클릭한다.
4. 사용자 변수의 `Path`를 선택하고 편집을 클릭한다.
5. 새로 만들기를 선택하여 다음 경로를 추가한다.

```
C:\ffmpeg\bin
```

6. 확인을 눌러 저장한다.
7. VS Code를 완전히 종료한 후 다시 실행한다.

### 4. 설치 확인

VS Code의 PowerShell 터미널에서 실행한다.

```
ffmpeg -version
```

```
ffprobe -version
```

정상 설치되었다면 FFmpeg 버전 정보가 출력된다.

Python에서도 확인할 수 있다.

```
import shutil# FFmpeg 실행 파일 경로 확인print("FFmpeg:", shutil.which("ffmpeg"))# FFprobe 실행 파일 경로 확인print("FFprobe:", shutil.which("ffprobe"))
```

정상 출력 예시는 다음과 같다.

```
FFmpeg: C:\ffmpeg\bin\ffmpeg.exe
FFprobe: C:\ffmpeg\bin\ffprobe.exe
```

`None`이 출력되면 PATH 설정을 다시 확인해야 한다.

### 5. yt_dlp과 FFmpeg 연동 확인

현재 사용 중인 `yt_lib.py`에서 다음과 같이 FFmpeg 설치 여부를 검사할 수 있다.

```
import shutilimport yt_dlp# FFmpeg 설치 여부 확인ffmpeg_path = shutil.which("ffmpeg")if ffmpeg_path is None:    raise RuntimeError(        "FFmpeg이 설치되어 있지 않거나 PATH에 등록되지 않았습니다."    )print("FFmpeg 경로:", ffmpeg_path)
```

추가로 `ffmpeg-python` 패키지를 사용하는 경우 다음 명령으로 설치한다.

```
python -m pip install ffmpeg-python yt-dlp
```

주의: `ffmpeg-python`은 FFmpeg 명령어를 Python에서 호출하는 라이브러리이다. `pip install ffmpeg-python`만 실행한다고 FFmpeg 실행 파일이 설치되는 것은 아니다.

### 6. 더 간단한 설치 방법

Windows에서 `winget`을 사용할 수 있다면 압축 파일을 직접 다운로드하지 않고 설치할 수도 있다.

```
winget install --id Gyan.FFmpeg -e
```

설치 후 VS Code를 재시작하고 다음 명령으로 확인한다.

```
ffmpeg -version
```

권장 방법: Windows 환경에서는 `winget` 설치가 간편하다. 설치가 끝나면 `shutil.which("ffmpeg")`로 Python 가상환경에서도 FFmpeg 실행 파일을 찾을 수 있는지 확인하면 된다.