import yt_dlp
import ffmpeg
import os
import shutil


def download_youtube_mp3(video_id: str = 'vHdKMkIH7qM', output_dir: str = "downloads"):
    # FFmpeg 설치 여부 확인
    if shutil.which("ffmpeg") is None:
        print("오류: FFmpeg이 설치되어 있지 않습니다. 설치 후 다시 시도하세요.")
        return None

    url = f"https://www.youtube.com/watch?v={video_id}"
    
    os.makedirs(output_dir, exist_ok=True)  # 다운로드 폴더 생성

    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f"{output_dir}/%(id)s.%(ext)s",
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
    }

    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)
            file_name = f"{info['id']}.mp3"
            file_path = os.path.join(output_dir, file_name)

        print(f"다운로드 완료! 파일 경로: {file_path}")
        return file_path
    except Exception as e:
        print(f"오류 발생: {e}")
        return None


def trim_audio_file(file_path: str, max_size: int = 24*1024*1024):
    '''오디오 파일을 24메가 크기로 자릅니다.'''
    file_size = os.path.getsize(file_path)  # 바이트 단위
    output_path = file_path
    if file_size >= max_size:
        # ffmpeg로 오디오 비트레이트 가져오기
        probe = ffmpeg.probe(file_path)
        bitrate = int(probe['format']['bit_rate'])  # 비트레이트 (bps)

        # 최대 24MB로 자를 수 있는 재생 시간 계산
        max_duration = max_size * 8 / bitrate  # 초 단위로 변환

        # 출력 파일 경로 설정
        output_path = file_path.replace(".mp3", "_trimmed.mp3")

        # ffmpeg로 파일 자르기
        ffmpeg.input(file_path, ss=0, t=max_duration).output(output_path, codec="copy").run(overwrite_output=True)
    return output_path

