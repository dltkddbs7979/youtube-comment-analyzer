# -*- coding: utf-8 -*-
"""CSV, JSON 내보내기. 작성자와 API 키는 넣지 않는다."""

from __future__ import annotations

import csv
import io
import json
import re

from src.textutil import excerpt

CSV_FIELDS = [
    "영상ID",
    "댓글원문",
    "정제텍스트",
    "주분류",
    "보조태그",
    "신뢰도",
    "근거",
    "좋아요수",
    "작성시각",
    "대댓글여부",
    "분류방식",
    "분류기버전",
    "사용자수정여부",
    "판단보류사유",
    "개인상황언급",
    "낮은신뢰도",
    "수집시각",
]


def _yn(value: bool) -> str:
    return "예" if value else "아니오"


def comment_row(record: dict) -> dict:
    return {
        "영상ID": record.get("video_id") or "",
        "댓글원문": record.get("text_raw") or "",
        "정제텍스트": record.get("text_analysis") or "",
        "주분류": record.get("label_name") or "",
        "보조태그": "; ".join(record.get("secondary_names") or []),
        "신뢰도": record.get("confidence", ""),
        "근거": " / ".join(record.get("evidence") or []),
        "좋아요수": record.get("like_count", 0),
        "작성시각": record.get("published_at") or "",
        "대댓글여부": _yn(bool(record.get("is_reply"))),
        "분류방식": record.get("classifier_name") or "",
        "분류기버전": record.get("classifier_version") or "",
        "사용자수정여부": _yn(bool(record.get("user_modified"))),
        "판단보류사유": record.get("hold_reason") or "",
        "개인상황언급": "; ".join(record.get("personal_terms") or []),
        "낮은신뢰도": _yn(bool(record.get("low_confidence"))),
        "수집시각": record.get("collected_at") or "",
    }


def excluded_row(record: dict) -> dict:
    row = comment_row(record)
    row["주분류"] = ""
    row["판단보류사유"] = record.get("exclude_reason") or record.get("hold_reason") or ""
    return row


def to_csv_bytes(rows: list[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=CSV_FIELDS, extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def analysis_csv_bytes(records: list[dict]) -> bytes:
    return to_csv_bytes([comment_row(record) for record in records])


def excluded_csv_bytes(excluded: list[dict], duplicates: list[dict]) -> bytes:
    rows = []
    for record in excluded:
        row = excluded_row(record)
        row["판단보류사유"] = record.get("exclude_reason") or ""
        rows.append(row)
    for record in duplicates:
        row = excluded_row(record)
        row["판단보류사유"] = "중복 댓글"
        rows.append(row)
    return to_csv_bytes(rows)


def _public_record(record: dict) -> dict:
    """작성자를 식별하는 필드는 내보내지 않는다."""
    return {
        "video_id": record.get("video_id"),
        "text_raw": record.get("text_raw"),
        "text_analysis": record.get("text_analysis"),
        "label": record.get("label_name"),
        "secondary": record.get("secondary_names") or [],
        "confidence": record.get("confidence"),
        "confidence_note": "검증된 확률이 아닌 규칙 참고 점수",
        "evidence": record.get("evidence") or [],
        "like_count": record.get("like_count"),
        "published_at": record.get("published_at"),
        "is_reply": bool(record.get("is_reply")),
        "classifier": record.get("classifier_name"),
        "classifier_version": record.get("classifier_version"),
        "user_modified": bool(record.get("user_modified")),
        "hold_reason": record.get("hold_reason") or "",
        "personal_terms": record.get("personal_terms") or [],
        "low_confidence": bool(record.get("low_confidence")),
        "collected_at": record.get("collected_at"),
        "exclude_reason": record.get("exclude_reason") or "",
    }


def build_json_document(bundle: dict) -> dict:
    meta = dict(bundle.get("meta") or {})
    for secret in ("api_key", "authorization", "YOUTUBE_API_KEY"):
        meta.pop(secret, None)
    summary = bundle.get("summary") or {}
    return {
        "notice": meta.get("notice") or "",
        "video": {
            "video_id": meta.get("video_id"),
            "video_url": meta.get("video_url"),
            "title": meta.get("video_title") or "",
            "source": meta.get("source"),
            "is_sample": bool(meta.get("is_sample")),
        },
        "collection": {
            "collected_at": meta.get("collected_at"),
            "pages": meta.get("pages"),
            "reply_pages": meta.get("reply_pages"),
            "api_comment_count": meta.get("api_comment_count"),
            "partial": bool(meta.get("partial")),
            "warning": meta.get("warning") or "",
        },
        "preprocess": bundle.get("report") or {},
        "classification": {
            "method": meta.get("classifier_name"),
            "version": meta.get("classifier_version"),
            "confidence_note": (summary.get("stats") or {}).get("confidence_note"),
            "external_transfer": bool(meta.get("external_transfer")),
        },
        "stats": summary.get("stats"),
        "keywords": {
            "overall": (summary.get("keywords") or {}).get("overall"),
            "by_category": (summary.get("keywords") or {}).get("by_category"),
            "collocations": (summary.get("keywords") or {}).get("collocations"),
            "title_discount": (summary.get("keywords") or {}).get("title_discount"),
        },
        "comparison": summary.get("comparison"),
        "replies": summary.get("replies"),
        "timeline": summary.get("timeline"),
        "research_answers": summary.get("answers"),
        "conclusion": summary.get("conclusion"),
        "limitations": summary.get("limitations"),
        "representative_criteria": (summary.get("representatives") or {}).get("criteria"),
        "comments": [_public_record(record) for record in bundle.get("records") or []],
        "excluded_comments": [_public_record(record) for record in bundle.get("excluded") or []],
        "duplicate_comments": [
            {"text_analysis": excerpt(record.get("text_analysis") or "", 180), "duplicate_of_kept": True}
            for record in bundle.get("duplicates") or []
        ],
    }


def analysis_json_bytes(bundle: dict) -> bytes:
    document = build_json_document(bundle)
    text = json.dumps(document, ensure_ascii=False, indent=2)
    if re.search(r"\\u[0-9a-fA-F]{4}", text):
        # ensure_ascii=False이면 한글은 그대로 둔다. 다른 이스케이프가 생기면 다시 확인한다.
        pass
    return text.encode("utf-8")


def file_stamp(collected_at: str, video_id: str) -> str:
    digits = re.sub(r"[^0-9]", "", collected_at or "")[:14]
    if len(digits) < 8:
        digits = "unknown-time"
    safe_id = re.sub(r"[^A-Za-z0-9_-]", "", video_id or "video")
    return f"{safe_id}_{digits}"
