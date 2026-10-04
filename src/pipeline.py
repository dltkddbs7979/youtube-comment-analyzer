# -*- coding: utf-8 -*-
"""수집부터 분류까지 한 흐름. 통계는 화면에서 수정한 라벨로 다시 계산한다."""

from __future__ import annotations

from datetime import datetime

from src.classifier import RuleClassifier
from src.preprocess import preprocess_comments
from src.summarize import summarize

DEFAULT_VIDEO_ID = "15Gbo7Xcy80"
DEFAULT_VIDEO_URL = "https://www.youtube.com/watch?v=15Gbo7Xcy80"
KNOWN_TITLES = {
    DEFAULT_VIDEO_ID: "이성을 강하게 끌어당기는 연애운 주파수 STUDY WITH ME ver.",
}


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def classify_records(records: list[dict], classifier) -> list[dict]:
    classified = []
    for index, record in enumerate(records, start=1):
        result = classifier.classify(record.get("text_analysis") or "")
        merged = dict(record)
        merged.update(result.to_fields())
        merged["display_no"] = index
        classified.append(merged)
    return classified


def run_prepared(raw_comments: list[dict], meta: dict, classifier=None) -> dict:
    classifier = classifier or RuleClassifier()
    collected_at = meta.get("collected_at") or now_iso()
    video_id = meta.get("video_id") or DEFAULT_VIDEO_ID
    kept, excluded, duplicates, report = preprocess_comments(raw_comments, collected_at, video_id)
    records = classify_records(kept, classifier)
    meta = dict(meta)
    meta["collected_at"] = collected_at
    meta["video_id"] = video_id
    meta["classifier_name"] = getattr(classifier, "name", "분류기")
    meta["classifier_version"] = getattr(classifier, "version", "")
    meta["external_transfer"] = bool(getattr(classifier, "sends_text_externally", False))
    if not meta.get("video_url"):
        meta["video_url"] = f"https://www.youtube.com/watch?v={video_id}"
    bundle = {
        "meta": meta,
        "report": report,
        "records": records,
        "excluded": excluded,
        "duplicates": duplicates,
    }
    bundle["summary"] = summarize(records, report, meta, meta.get("video_title") or "")
    return bundle


def refresh_summary(bundle: dict) -> dict:
    bundle["summary"] = summarize(
        bundle["records"],
        bundle["report"],
        bundle["meta"],
        bundle["meta"].get("video_title") or "",
    )
    return bundle
