# -*- coding: utf-8 -*-
"""모의 댓글 로더. 실제 수집 결과와 섞이지 않도록 안내 문구를 함께 둔다."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SAMPLE_PATH = ROOT / "data" / "sample_comments.json"


def load_sample_file() -> dict:
    return json.loads(SAMPLE_PATH.read_text(encoding="utf-8"))


def load_sample_comments() -> tuple[list[dict], str, str]:
    data = load_sample_file()
    comments = []
    for item in data["comments"]:
        comments.append({key: value for key, value in item.items() if not str(key).startswith("_")})
    notice = data.get("notice") or "모의 데이터입니다. 실제 유튜브 댓글이 아닙니다."
    title = data.get("video_title") or ""
    return comments, notice, title
