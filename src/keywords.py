# -*- coding: utf-8 -*-
"""한국어 키워드.

kiwipiepy로 명사·동사·형용사를 뽑는다.
설치되어 있지 않거나 불러오지 못하면 어절 기반으로 대체하고, 그 사실을 결과에 남긴다.
영상 제목에 들어 있는 단어는 삭제하지 않고 점수를 낮춘다.
"""

from __future__ import annotations

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
STOPWORD_PATH = ROOT / "data" / "stopwords.txt"

CONTENT_TAGS = {"NNG", "NNP", "VV", "VA", "XR", "SL"}
POS_KO = {
    "NNG": "명사",
    "NNP": "고유명사",
    "VV": "동사",
    "VA": "형용사",
    "XR": "어근",
    "SL": "외국어",
    "UNK": "어절",
}

_KIWI = None
_KIWI_TRIED = False
_KIWI_ERROR = ""


def load_stopwords(path: Path | None = None) -> set[str]:
    target = path or STOPWORD_PATH
    words: set[str] = set()
    if not target.is_file():
        return words
    for line in target.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        words.add(stripped.casefold())
    return words


def analyzer_status() -> dict:
    kiwi, error = _get_kiwi()
    if kiwi is not None:
        return {
            "backend": "kiwipiepy",
            "ok": True,
            "message": "kiwipiepy로 명사, 동사, 형용사를 추출했습니다.",
        }
    return {
        "backend": "어절 대체",
        "ok": False,
        "message": (
            "형태소 분석기를 사용하지 못했습니다. "
            f"{error} 2글자 이상의 어절로 키워드를 대체 계산했습니다. "
            "더 안정적인 품사 구분이 필요하면 pip install kiwipiepy 후 다시 실행하세요. Java는 필요하지 않습니다."
        ),
    }


def _get_kiwi():
    global _KIWI, _KIWI_TRIED, _KIWI_ERROR
    if _KIWI_TRIED:
        return _KIWI, _KIWI_ERROR
    _KIWI_TRIED = True
    try:
        from kiwipiepy import Kiwi

        _KIWI = Kiwi()
    except ImportError:
        _KIWI = None
        _KIWI_ERROR = "kiwipiepy가 설치되어 있지 않습니다."
    except Exception:
        _KIWI = None
        _KIWI_ERROR = "kiwipiepy를 불러오지 못했습니다."
    return _KIWI, _KIWI_ERROR


def tokenize(text: str) -> list[tuple[str, str]]:
    kiwi, _error = _get_kiwi()
    if kiwi is None:
        return _fallback_tokens(text)
    tokens: list[tuple[str, str]] = []
    try:
        analyzed = kiwi.tokenize(text)
    except Exception:
        return _fallback_tokens(text)
    for token in analyzed:
        tag = getattr(token, "tag", "")
        if tag not in CONTENT_TAGS:
            continue
        lemma = getattr(token, "lemma", "") or getattr(token, "form", "")
        lemma = str(lemma).strip()
        if tag == "VV" and lemma.endswith("다") and len(lemma) > 2:
            lemma = lemma[:-1]
        if len(lemma) < 2:
            continue
        tokens.append((lemma, tag))
    return tokens


def _fallback_tokens(text: str) -> list[tuple[str, str]]:
    words = re.findall(r"[가-힣]{2,}|[A-Za-z]{2,}", text or "")
    return [(word, "UNK") for word in words]


def title_words(title: str) -> set[str]:
    return {lemma.casefold() for lemma, _tag in tokenize(title or "")}


def keyword_table(texts: list[str], stopwords: set[str], title_token_set: set[str], min_count: int = 2, limit: int = 12) -> dict:
    counts: dict[str, dict] = {}
    for text in texts:
        seen_in_text: set[str] = set()
        for lemma, tag in tokenize(text):
            key = lemma.casefold()
            if key in stopwords or key in seen_in_text:
                continue
            seen_in_text.add(key)
            slot = counts.setdefault(key, {"word": lemma, "pos": POS_KO.get(tag, tag), "count": 0})
            slot["count"] += 1
            if len(lemma) > len(slot["word"]):
                slot["word"] = lemma

    rows = []
    for key, slot in counts.items():
        discounted = key in title_token_set
        score = round(slot["count"] * (0.35 if discounted else 1), 2)
        if slot["count"] < min_count:
            continue
        rows.append(
            {
                "단어": slot["word"],
                "품사": slot["pos"],
                "빈도": slot["count"],
                "반영점수": score,
                "비고": "제목에 포함된 단어라 반영 점수를 0.35배로 낮춤" if discounted else "",
            }
        )
    rows.sort(key=lambda row: (-row["반영점수"], -row["빈도"], row["단어"]))
    note = ""
    if not rows and counts:
        for key, slot in counts.items():
            discounted = key in title_token_set
            score = round(slot["count"] * (0.35 if discounted else 1), 2)
            rows.append(
                {
                    "단어": slot["word"],
                    "품사": slot["pos"],
                    "빈도": slot["count"],
                    "반영점수": score,
                    "비고": "등장 횟수가 적어 참고용입니다."
                    + (" 제목어 할인을 적용했습니다." if discounted else ""),
                }
            )
        rows.sort(key=lambda row: (-row["반영점수"], -row["빈도"], row["단어"]))
        note = "반복된 단어가 적어 1회 등장 단어도 참고용으로 보여 줍니다."
    return {"rows": rows[:limit], "note": note}


def collocations(texts: list[str], stopwords: set[str], limit: int = 8) -> list[dict]:
    pair_counts: dict[tuple[str, str], int] = {}
    for text in texts:
        tokens = []
        seen = set()
        for lemma, _tag in tokenize(text):
            key = lemma.casefold()
            if key in stopwords or key in seen:
                continue
            seen.add(key)
            tokens.append(lemma)
        for left, right in zip(tokens, tokens[1:]):
            pair = (left, right)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
    rows = [
        {"표현": f"{left} + {right}", "빈도": count}
        for (left, right), count in pair_counts.items()
        if count >= 2
    ]
    rows.sort(key=lambda row: (-row["빈도"], row["표현"]))
    return rows[:limit]
