# -*- coding: utf-8 -*-
"""텍스트 정리, 개인정보 가림, 중복 키."""

from __future__ import annotations

import html
import re
import unicodedata

URL_RE = re.compile(r"(https?://\S+|www\.\S+)", re.IGNORECASE)
PHONE_RE = re.compile(r"(?<!\d)(?:\+82[-\s]?)?0?1[016789][-\s.]?\d{3,4}[-\s.]?\d{4}(?!\d)")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
RRN_RE = re.compile(r"(?<!\d)\d{6}\s*-\s*[1-4]\d{6}(?!\d)")
TAG_RE = re.compile(r"<[^>]+>")
BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)
EMOJI_RE = re.compile(r"[\U0001F300-\U0001FAFF\u2600-\u27BF\uFE0F]")
HANGUL_RE = re.compile(r"[가-힣]")
LAUGH_RE = re.compile(r"[ㅋㅎ]")


def normalize_text(value: str) -> str:
    text = "" if value is None else str(value)
    text = BR_RE.sub("\n", text)
    text = TAG_RE.sub("", text)
    text = html.unescape(text)
    text = text.replace("\xa0", " ")
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[\u200b\u200c\u200d\ufeff]", "", text)
    text = text.replace("\u110f", "ㅋ").replace("\u1112", "ㅎ")
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def mask_pii(text: str) -> str:
    masked = EMAIL_RE.sub("[이메일]", text)
    masked = PHONE_RE.sub("[전화번호]", masked)
    masked = RRN_RE.sub("[식별번호]", masked)
    return masked


def replace_urls(text: str) -> str:
    return URL_RE.sub("[URL]", text)


def duplicate_key(text: str) -> str:
    collapsed = re.sub(r"\s+", " ", text).strip().casefold()
    return collapsed


def meaningful_length(text: str) -> int:
    """웃음 기호를 뺀 한글 음절 수. 짧은 감탄을 가려 낼 때 쓴다."""
    hangul = HANGUL_RE.findall(text)
    laughs = LAUGH_RE.findall(text)
    return max(0, len(hangul) - len(laughs))


def has_emoji(text: str) -> bool:
    return EMOJI_RE.search(text) is not None


def has_textual_content(text: str) -> bool:
    if re.search(r"[A-Za-z가-힣0-9ㅋㅎ]", text):
        return True
    return has_emoji(text)


def excerpt(text: str, limit: int = 80) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1] + "…"
