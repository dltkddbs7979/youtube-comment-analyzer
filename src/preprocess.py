# -*- coding: utf-8 -*-
"""댓글 전처리. 원문과 분석용 텍스트를 따로 둔다."""

from __future__ import annotations

import re

from src.textutil import (
    duplicate_key,
    has_textual_content,
    mask_pii,
    normalize_text,
    replace_urls,
)

EXCLUSION_CRITERIA = [
    "홍보·스팸 의심: 텔레그램, 오픈채팅, 구독 유도, 홍보 문구가 있는 댓글",
    "무의미한 기호만 있음: 한글, 영문, 숫자, 이모지가 없는 댓글",
    "무의미한 반복 문자: 같은 문자가 30회 이상 반복되고 내용이 거의 없는 댓글",
    "분석할 텍스트 없음: 정리 후 빈 문자열",
    "완전히 같은 댓글은 제외가 아니라 중복으로 따로 센다. 먼저 작성된 댓글 하나만 남긴다.",
]


def _spam_reason(text: str, raw_with_url: str) -> str | None:
    del raw_with_url
    if not text:
        return "분석할 텍스트 없음"
    if not has_textual_content(text):
        return "무의미한 기호만 있음"
    if re.search(r"(.)\1{29,}", text):
        return "무의미한 반복 문자"
    promo = re.search(
        r"텔레그램|텔레\s*그램|t\.me/|오픈채팅|오픈\s*채팅|카카오\s*오픈|bit\.ly/|홍보합니다|광고\s*문의|구독\s*(?:하고|눌러|부탁)",
        text,
        re.IGNORECASE,
    )
    if promo:
        return "홍보·스팸 의심"
    return None


def prepare_comment(raw: dict, collected_at: str, video_id: str) -> dict:
    original = normalize_text(raw.get("text") or raw.get("text_raw") or "")
    masked = mask_pii(original)
    return {
        "comment_id": str(raw.get("comment_id") or ""),
        "parent_id": raw.get("parent_id") or None,
        "video_id": video_id,
        "text_raw": masked,
        "text_analysis": replace_urls(masked),
        "published_at": raw.get("published_at") or "",
        "like_count": int(raw.get("like_count") or 0),
        "is_reply": bool(raw.get("is_reply")),
        "collected_at": collected_at,
    }


def preprocess_comments(raw_comments: list[dict], collected_at: str, video_id: str) -> tuple[list[dict], list[dict], list[dict], dict]:
    prepared = [prepare_comment(item, collected_at, video_id) for item in raw_comments]
    prepared.sort(key=lambda item: (item.get("published_at") or "", item.get("comment_id") or ""))

    seen: dict[str, dict] = {}
    duplicates: list[dict] = []
    for item in prepared:
        key = duplicate_key(item["text_analysis"])
        if not key:
            continue
        if key in seen:
            copy = dict(item)
            copy["duplicate_of"] = seen[key]["comment_id"]
            duplicates.append(copy)
        else:
            seen[key] = item

    unique = [item for item in prepared if duplicate_key(item["text_analysis"]) in seen and seen[duplicate_key(item["text_analysis"])] is item]
    # 빈 문자열은 seen에 넣지 않았으므로 아래에서 제외 사유로 처리한다.
    empty_items = [item for item in prepared if not duplicate_key(item["text_analysis"])]

    kept: list[dict] = []
    excluded: list[dict] = []
    for item in empty_items:
        copy = dict(item)
        copy["exclude_reason"] = "분석할 텍스트 없음"
        excluded.append(copy)

    for item in unique:
        reason = _spam_reason(item["text_analysis"], item["text_raw"])
        if reason:
            copy = dict(item)
            copy["exclude_reason"] = reason
            excluded.append(copy)
        else:
            kept.append(item)

    reasons: dict[str, int] = {}
    for item in excluded:
        reasons[item["exclude_reason"]] = reasons.get(item["exclude_reason"], 0) + 1

    report = {
        "raw_count": len(prepared),
        "duplicate_count": len(duplicates),
        "excluded_count": len(excluded),
        "analyzed_count": len(kept),
        "exclusion_reasons": reasons,
        "criteria": EXCLUSION_CRITERIA,
    }
    return kept, excluded, duplicates, report
