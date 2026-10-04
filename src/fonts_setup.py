# -*- coding: utf-8 -*-
"""차트용 한글 글꼴.

프로젝트에 포함한 IBM Plex Sans KR을 우선 사용한다.
운영체제 글꼴만으로 차트를 그리지 않는다.
글꼴 파일이 없으면 깨진 차트를 만들지 않고 오류를 반환한다.
"""

from __future__ import annotations

import base64
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BUNDLED_FONT = ROOT / "assets" / "fonts" / "IBMPlexSansKR-Regular.ttf"
BUNDLED_FONT_NAME = "IBM Plex Sans KR"
BUNDLED_LICENSE = ROOT / "assets" / "fonts" / "OFL.txt"

SYSTEM_FONT_CANDIDATES = [
    Path(r"C:\Windows\Fonts\malgun.ttf"),
    Path(r"C:\Windows\Fonts\malgunbd.ttf"),
    Path("/System/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/Library/Fonts/AppleSDGothicNeo.ttc"),
    Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc"),
    Path("/usr/share/fonts/truetype/nanum/NanumGothic.ttf"),
]


def bundled_font_path() -> Path | None:
    if BUNDLED_FONT.is_file() and BUNDLED_FONT.stat().st_size > 10_000:
        return BUNDLED_FONT
    return None


def chart_font_path() -> Path | None:
    """차트 글리프를 그릴 글꼴. 번들 글꼴이 있으면 그것만 사용한다."""
    bundled = bundled_font_path()
    if bundled is not None:
        return bundled
    for candidate in SYSTEM_FONT_CANDIDATES:
        if candidate.is_file():
            return candidate
    return None


def font_status() -> dict:
    system = [str(path) for path in SYSTEM_FONT_CANDIDATES if path.is_file()]
    bundled = bundled_font_path()
    return {
        "bundled_exists": bundled is not None,
        "bundled_name": BUNDLED_FONT_NAME,
        "bundled_path": str(BUNDLED_FONT),
        "license_exists": BUNDLED_LICENSE.is_file(),
        "system_fonts": system,
        "chart_font": str(chart_font_path() or ""),
        "ready": chart_font_path() is not None,
    }


def missing_font_message() -> tuple[str, str, str]:
    return (
        "한글 차트 글꼴을 찾지 못했습니다",
        "프로젝트의 IBM Plex Sans KR 파일도, 사용할 수 있는 한글 글꼴도 없습니다. 깨진 차트는 표시하지 않습니다.",
        "assets/fonts/IBMPlexSansKR-Regular.ttf 파일이 있는지 확인하세요. "
        "파일이 없다면 README의 글꼴 안내를 따라 같은 OFL 글꼴을 다시 두세요. "
        "그 전까지는 아래 표로 같은 수치를 확인할 수 있습니다.",
    )


def ui_font_css() -> str:
    """화면 본문용 CSS. 윈도우에서는 맑은 고딕을 명시하고, 한글 글꼴이 없으면 번들 글꼴을 넣는다."""
    stack = (
        '"Malgun Gothic", "맑은 고딕", "Apple SD Gothic Neo", '
        '"Noto Sans KR", "Noto Sans CJK KR", "NanumGothic", '
        f'"{BUNDLED_FONT_NAME}", sans-serif'
    )
    embed = ""
    if bundled_font_path() is not None and not _has_known_ui_font():
        encoded = base64.b64encode(BUNDLED_FONT.read_bytes()).decode("ascii")
        embed = f"""
        @font-face {{
            font-family: "{BUNDLED_FONT_NAME}";
            src: url("data:font/truetype;base64,{encoded}") format("truetype");
            font-weight: 400;
            font-style: normal;
            font-display: swap;
        }}
        """
    return embed + f"""
    html, body, .stApp, [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li, label, input, textarea {{
        font-family: {stack};
    }}
    """


def _has_known_ui_font() -> bool:
    return any(path.is_file() for path in SYSTEM_FONT_CANDIDATES)
