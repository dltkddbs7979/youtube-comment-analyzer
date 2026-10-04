# -*- coding: utf-8 -*-
"""모의 데이터로 한글 차트를 파일로 저장한다."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.charts import render_category_chart
from src.pipeline import run_prepared
from src.sample_data import load_sample_comments


def main() -> Path:
    comments, notice, title = load_sample_comments()
    bundle = run_prepared(
        comments,
        {
            "video_id": "15Gbo7Xcy80",
            "video_title": title,
            "is_sample": True,
            "notice": notice,
            "source": "sample",
            "collected_at": "2026-04-01T12:00:00+09:00",
        },
    )
    rows = bundle["summary"]["stats"]["rows"]
    _svg, png = render_category_chart(rows, bundle["summary"]["stats"]["total"])
    output = ROOT / "output"
    output.mkdir(exist_ok=True)
    path = output / "korean_chart_check.png"
    path.write_bytes(png)
    print(path)
    return path


if __name__ == "__main__":
    main()
