from pathlib import Path
import re

def clean_markdown(RAW_MD_PATH: Path, CLEAN_MD_PATH: Path):
    
    raw_text = RAW_MD_PATH.read_text(encoding="utf-8")

    clean_text = raw_text

    # 1. 이미지 생략 문구 제거
    clean_text = re.sub(
        r"\*\*==> picture \[[^\]]+\] intentionally omitted <==\*\*",
        "",
        clean_text
    )

    # 2. 그림 텍스트 시작/끝 표시 제거
    clean_text = clean_text.replace("**----- Start of picture text -----**<br>", "")
    clean_text = clean_text.replace("**----- End of picture text -----**<br>", "")

    # 3. HTML 줄바꿈 정리
    clean_text = clean_text.replace("<br>", "\n")

    # 4. 특수 bullet 정리
    clean_text = clean_text.replace("Ÿ", "-")

    # 5. 반복되는 문서 제목/페이지 헤더 일부 정리
    clean_text = re.sub(
        r"\n?\d+\s+토픽 분석을 통한 AI 주요 트렌드 및 2026 전망\s*\n?",
        "\n",
        clean_text
    )

    clean_text = re.sub(
        r"\n?토픽 분석을 통한 AI 주요 트렌드 및 2026 전망\s+\d+\s*\n?",
        "\n",
        clean_text
    )

    # 6. |||| 문자열이 포함된 라인 삭제
    clean_text = "\n".join(
        line for line in clean_text.splitlines()
        if "||||" not in line
    )

    # 7. 과도한 빈 줄 정리
    clean_text = re.sub(r"\n{3,}", "\n\n", clean_text)

    CLEAN_MD_PATH.write_text(clean_text.strip(), encoding="utf-8")

    print(f"정제된 Markdown 저장 완료: {CLEAN_MD_PATH}")
    print(clean_text[:1000])

if __name__ == "__main__":
    RAW_MD_PATH = Path("data/AI@Data_Report_2026_전망_251223(최종).md")
    CLEAN_MD_PATH = Path("data/AI@Data_Report_CLEANED.md")

    clean_markdown(RAW_MD_PATH, CLEAN_MD_PATH)
    