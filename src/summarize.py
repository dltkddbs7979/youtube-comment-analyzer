# -*- coding: utf-8 -*-
"""통계, 대표 댓글, 연구 질문 답, 결론.

효능이나 인과를 단정하는 문장은 만들지 않는다.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime
import math
import re

from src.categories import CATEGORIES, CATEGORY_ORDER, label_text
from src.classifier import CONFIDENCE_NOTE
from src.keywords import collocations, keyword_table, load_stopwords, title_words
from src.textutil import excerpt

REPRESENTATIVE_NOTE = (
    "대표 댓글은 좋아요 수만으로 고르지 않습니다. "
    "해당 유형의 참고 점수, 유형의 특징이 드러나는 분량, 이미 고른 댓글과의 내용 차이를 함께 보고 "
    "최대 3건을 고릅니다. 좋아요 수는 보조 참고입니다."
)
REPRESENTATIVE_LIMIT = (
    "각 댓글은 작성자의 경험이나 주장을 보여 주는 사례일 뿐, "
    "전체 시청자나 주파수의 효과를 대표하지 않습니다."
)

LIMITATIONS = [
    "분석 대상은 댓글을 남긴 작성자입니다. 영상을 보기만 한 사람, 댓글을 쓰지 않은 사람의 생각으로 일반화할 수 없습니다.",
    "댓글은 자기선택 편향이 있습니다. 경험을 말하고 싶거나 불만을 말한 사람이 더 남을 수 있습니다.",
    "효과를 경험했다고 서술한 댓글은 작성자의 인식이나 주장입니다. 주파수가 연락, 재회, 연애를 일으켰다는 증거가 아닙니다.",
    "시간 순서나 상관처럼 보이는 서술도 인과관계로 해석하지 않습니다.",
    "삭제된 댓글, 비공개 댓글, 검토 대기 댓글, API가 반환하지 않은 대댓글은 포함되지 않습니다.",
    "영상 하나의 공개 댓글만 다루므로 다른 영상이나 다른 시점으로 넓혀 말할 수 없습니다.",
    "규칙 기반 분류의 참고 점수는 검증된 정확도나 확률이 아닙니다. 판단 보류와 낮은 참고 점수는 사람이 확인해야 합니다.",
    "판단이 어려운 댓글은 가장 가까운 유형으로 억지 배정하지 않고 판단 보류로 둡니다.",
]

HOPE_MARKERS = ["왔으면", "좋겠", "바라", "기도", "기원", "희망", "되길", "오길", "제발"]
FORBIDDEN_CONCLUSION = [
    "효과가 입증",
    "입증되었",
    "성사시켰다",
    "실제로 효과가 있다",
    "인과관계가 확인",
    "시청자 전체의 의견",
]


def _percent(count: int, total: int) -> float:
    if total <= 0:
        return 0.0
    return round(count * 100 / total, 1)


def category_stats(records: list[dict]) -> dict:
    total = len(records)
    counts = Counter(record["label"] for record in records)
    rows = []
    for key in CATEGORY_ORDER:
        count = counts.get(key, 0)
        rows.append(
            {
                "id": key,
                "유형": CATEGORIES[key]["name"],
                "기호": CATEGORIES[key]["symbol"],
                "라벨": label_text(key),
                "색": CATEGORIES[key]["color"],
                "댓글수": count,
                "비율": _percent(count, total),
            }
        )
    low = sum(1 for record in records if record.get("low_confidence"))
    modified = sum(1 for record in records if record.get("user_modified"))
    return {
        "total": total,
        "rows": rows,
        "hold": counts.get("hold", 0),
        "low_confidence": low,
        "modified": modified,
        "denominator": "중복 제거와 홍보·무의미 댓글 제외 이후의 분류 대상 댓글. 판단 보류를 포함합니다.",
        "confidence_note": CONFIDENCE_NOTE,
    }


def _parse_dt(value: str) -> datetime | None:
    if not value:
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def timeline(records: list[dict]) -> dict:
    weeks: dict[str, int] = defaultdict(int)
    for record in records:
        moment = _parse_dt(record.get("published_at") or "")
        if moment is None:
            continue
        start = moment.date().fromordinal(moment.date().toordinal() - moment.date().weekday())
        weeks[start.isoformat()] += 1
    rows = [{"주시작": key, "댓글수": weeks[key]} for key in sorted(weeks)]
    note = "작성 시각의 분포이며, 효과의 변화나 인과를 보여 주지 않습니다."
    if len(rows) < 2:
        note = "서로 다른 시점이 부족해 시간 흐름으로 해석하지 않습니다. " + note
    return {"rows": rows, "note": note}


def reply_overview(records: list[dict]) -> dict:
    by_id = {record["comment_id"]: record for record in records if record.get("comment_id")}
    pairs = []
    for record in records:
        parent_id = record.get("parent_id")
        if not record.get("is_reply") or not parent_id or parent_id not in by_id:
            continue
        parent = by_id[parent_id]
        pairs.append(
            {
                "부모유형": parent["label_name"],
                "답글유형": record["label_name"],
            }
        )
    counter = Counter((item["부모유형"], item["답글유형"]) for item in pairs)
    rows = [
        {"부모 댓글 유형": parent, "답글 유형": reply, "짝 수": count}
        for (parent, reply), count in sorted(counter.items(), key=lambda item: (-item[1], item[0][0], item[0][1]))
    ]
    reply_count = sum(1 for record in records if record.get("is_reply"))
    if not rows:
        note = "부모 댓글과 연결된 답글 짝이 없어, 이용자 간 반응을 해석하지 않습니다."
    elif len(pairs) < 5:
        note = f"연결된 답글 짝이 {len(pairs)}건뿐입니다. 이 수로 댓글 상호작용의 전형을 말하지 않습니다."
    else:
        note = "아래 표는 수집된 짝에서 관찰된 횟수입니다. 전체 대화 양상으로 일반화하지 않습니다."
    return {"reply_count": reply_count, "pair_count": len(pairs), "rows": rows, "note": note}


def _bigrams(text: str) -> set[str]:
    compact = re.sub(r"\s+", "", text or "")
    return {compact[index : index + 2] for index in range(len(compact) - 1)}


def _representative_score(record: dict) -> float:
    length = len(record.get("text_analysis") or "")
    length_score = min(length, 140) / 140
    likes = math.log1p(max(0, int(record.get("like_count") or 0)))
    like_score = min(likes / math.log1p(500), 1)
    confidence = float(record.get("confidence") or 0)
    if record.get("user_modified"):
        confidence = min(confidence, 0.45)
    return confidence * 0.5 + length_score * 0.4 + like_score * 0.1


def representatives(records: list[dict], per_category: int = 3) -> dict:
    chosen = {}
    for key in CATEGORY_ORDER:
        if key == "hold":
            continue
        pool = [record for record in records if record["label"] == key]
        pool.sort(key=_representative_score, reverse=True)
        picked = []
        used_grams: list[set[str]] = []
        for record in pool:
            grams = _bigrams(record.get("text_analysis") or "")
            if any(grams and used and len(grams & used) / len(grams | used) > 0.72 for used in used_grams):
                continue
            picked.append(record)
            used_grams.append(grams)
            if len(picked) >= per_category:
                break
        chosen[key] = picked
    hold_pool = [record for record in records if record["label"] == "hold"]
    hold_pool.sort(key=lambda record: (record.get("hold_reason") or "", -len(record.get("text_analysis") or "")))
    hold_examples = []
    seen_reasons = set()
    for record in hold_pool:
        reason = record.get("hold_reason") or "보류"
        if reason in seen_reasons:
            continue
        seen_reasons.add(reason)
        hold_examples.append(record)
        if len(hold_examples) >= 3:
            break
    return {
        "by_category": chosen,
        "hold_examples": hold_examples,
        "criteria": REPRESENTATIVE_NOTE,
        "limit": REPRESENTATIVE_LIMIT,
    }


def comparison_rows(records: list[dict], per_category_keywords: dict, stats: dict) -> list[dict]:
    rows = []
    stat_by_id = {row["id"]: row for row in stats["rows"]}
    for key in CATEGORY_ORDER:
        subset = [record for record in records if record["label"] == key]
        replies = sum(1 for record in subset if record.get("is_reply"))
        personal = sum(1 for record in subset if record.get("personal_context"))
        if subset:
            avg_conf = round(sum(float(record.get("confidence") or 0) for record in subset) / len(subset), 2)
            avg_len = round(sum(len(record.get("text_analysis") or "") for record in subset) / len(subset), 1)
        else:
            avg_conf = 0
            avg_len = 0
        words = [row["단어"] for row in per_category_keywords.get(key, {}).get("rows", [])[:3]]
        rows.append(
            {
                "유형": stat_by_id[key]["라벨"],
                "댓글수": stat_by_id[key]["댓글수"],
                "비율": stat_by_id[key]["비율"],
                "대댓글수": replies,
                "개인상황언급": personal,
                "평균참고점수": avg_conf,
                "평균글자수": avg_len,
                "상위키워드": ", ".join(words) if words else "—",
            }
        )
    return rows


def _hope_markers(records: list[dict]) -> list[str]:
    texts = [record.get("text_analysis") or "" for record in records if record["label"] == "hope"]
    return [marker for marker in HOPE_MARKERS if any(marker in text for text in texts)]


def _count(records: list[dict], label: str) -> int:
    return sum(1 for record in records if record["label"] == label)


def research_answers(records: list[dict], stats: dict, keywords: dict, replies: dict, meta: dict) -> list[dict]:
    total = stats["total"]
    hedge = ""
    if meta.get("is_sample"):
        hedge = "이 답은 모의 데이터로 계산한 예시이며, 실제 댓글 분석 결과가 아닙니다. "
    elif total < 30:
        hedge = "분류 대상이 적어 이 자료만으로는 판단하기 어렵습니다. "

    def count_pct(label: str) -> tuple[int, float]:
        count = _count(records, label)
        return count, _percent(count, total)

    hope_n, hope_p = count_pct("hope")
    effect_n, effect_p = count_pct("effect")
    usage_n, usage_p = count_pct("usage")
    criticism_n, criticism_p = count_pct("criticism")
    empathy_n, empathy_p = count_pct("empathy")
    humor_n, humor_p = count_pct("humor")
    personal_n = sum(1 for record in records if record.get("personal_context"))
    personal_p = _percent(personal_n, total)
    hope_words = [row["단어"] for row in keywords["by_category"].get("hope", {}).get("rows", [])[:5]]
    criticism_words = [row["단어"] for row in keywords["by_category"].get("criticism", {}).get("rows", [])[:5]]
    markers = _hope_markers(records)
    personal_terms = []
    for record in records:
        for term in record.get("personal_terms") or []:
            if term not in personal_terms:
                personal_terms.append(term)

    purpose_words = ", ".join(hope_words) if hope_words else "반복된 핵심어가 충분하지 않습니다"
    marker_text = ", ".join(markers) if markers else "소망 표현이 뚜렷이 반복되지는 않았습니다"

    if total == 0:
        empty = hedge + "분류 대상 댓글이 없어 이 질문에는 답할 수 없습니다."
        questions = [
            "댓글에서 드러나는 이용 목적이나 기대는 무엇인가?",
            "주파수 영상을 들은 뒤 효과를 경험했다고 말하는 댓글은 얼마나 나타나는가?",
            "효과를 기대하거나 희망하는 댓글은 어떤 내용으로 표현되는가?",
            "사용 방법을 묻거나 다른 댓글과 경험을 주고받는 반응은 어떻게 나타나는가?",
            "효과를 회의하거나 콘텐츠를 비판하는 댓글은 어떤 특징을 보이는가?",
            "영상이 댓글 작성자의 개인적 상황이나 경험과 연결되어 표현되는 양상은 무엇인가?",
        ]
        return [{"question": question, "answer": empty} for question in questions]

    q1 = (
        f"{hedge}분류 대상 {total}건 가운데 기대·희망형은 {hope_n}건({hope_p}%), "
        f"사용 방법 질문형은 {usage_n}건({usage_p}%), 공감·추천형은 {empathy_n}건({empathy_p}%)입니다. "
        f"기대·희망형에서 자주 나타난 단어는 {purpose_words}입니다. "
        "댓글을 남긴 사람들의 표현에서는 연애나 연락처럼 원하는 일을 바라며 영상을 듣는 목적이 관찰됩니다. "
        "이것은 댓글 작성자의 표현이지, 시청자 전체의 이용 목적은 아닙니다."
    )
    if effect_n == 0:
        q2 = (
            f"{hedge}분류 대상 {total}건에서 효과를 경험했다고 서술한 댓글은 관찰되지 않았습니다. "
            "이 자료만으로는 그런 서술이 얼마나 있는지를 말하기 어렵습니다. "
            "설령 그런 댓글이 있어도 작성자의 주장이지 주파수의 효과를 증명하지는 않습니다."
        )
    else:
        q2 = (
            f"{hedge}효과를 경험했다고 서술한 댓글은 {effect_n}건으로, 분류 대상 {total}건의 {effect_p}%입니다. "
            "여기서 말하는 경험은 작성자가 연락, 재회, 연애 성사처럼 원하던 일이 일어났다고 적은 내용입니다. "
            "주파수가 그 일을 일으켰다는 뜻으로 읽지 않습니다."
        )
    q3 = (
        f"{hedge}기대·희망형은 {hope_n}건({hope_p}%)입니다. "
        f"이 유형의 댓글에서 확인된 소망 표현은 {marker_text}입니다. "
        "이미 결과가 일어났다고 말하기보다, 일어나기를 바라거나 들겠다고 다짐하는 문장이 중심입니다."
    )
    if replies["pair_count"] == 0:
        interaction = "부모 댓글과 연결된 답글 짝은 확인되지 않았습니다."
    else:
        observed = ", ".join(
            f"{row['부모 댓글 유형']}에 대한 {row['답글 유형']} {row['짝 수']}건" for row in replies["rows"][:4]
        )
        interaction = f"연결된 짝 {replies['pair_count']}건에서 {observed}이 관찰됩니다. {replies['note']}"
    if usage_n == 0:
        usage_sentence = "사용 방법을 묻는 댓글은 이 자료에서 관찰되지 않았습니다."
    else:
        usage_sentence = "질문으로 확인된 내용은 이어폰·스피커, 반복 횟수, 듣는 시점처럼 이용 방법에 머뭅니다."
    q4 = (
        f"{hedge}사용 방법을 묻는 댓글은 {usage_n}건({usage_p}%)이고, 대댓글은 {replies['reply_count']}건입니다. "
        f"{interaction} {usage_sentence}"
    )
    if criticism_n == 0 and humor_n == 0:
        q5 = hedge + "회의·비판형과 유머·조롱형 댓글이 없어, 이 자료만으로는 비판의 특징을 말하기 어렵습니다."
    else:
        word_text = ", ".join(criticism_words) if criticism_words else "반복 단어가 충분하지 않습니다"
        q5 = (
            f"{hedge}회의·비판형은 {criticism_n}건({criticism_p}%), 유머·조롱형은 {humor_n}건({humor_p}%)입니다. "
            f"비판 댓글에서 자주 나타난 단어는 {word_text}입니다. "
            "관찰된 비판은 근거 부족, 플라시보, 광고성, 효과가 없었다는 자기 보고처럼 콘텐츠의 주장을 문제 삼는 표현입니다. "
            "이 비율을 시청자 다수의 판단으로 확대하지 않습니다."
        )
    if personal_n == 0:
        q6 = hedge + "개인적 상황으로 표시할 표현이 거의 없어, 이 자료만으로는 연결 양상을 말하기 어렵습니다."
    else:
        term_text = ", ".join(personal_terms[:8]) if personal_terms else "관계·이별·짝사랑 관련 표현"
        q6 = (
            f"{hedge}개인 상황 표현이 있는 댓글은 {personal_n}건({personal_p}%)입니다. "
            f"확인된 표현은 {term_text}입니다. "
            "영상 속 주파수를 작성자 자신의 연애 경험이나 기다리는 관계와 연결해 말하는 사례가 관찰됩니다. "
            "댓글을 쓰지 않은 시청자에게 같은 사정이 있다고 볼 수는 없습니다."
        )
    questions = [
        "댓글에서 드러나는 이용 목적이나 기대는 무엇인가?",
        "주파수 영상을 들은 뒤 효과를 경험했다고 말하는 댓글은 얼마나 나타나는가?",
        "효과를 기대하거나 희망하는 댓글은 어떤 내용으로 표현되는가?",
        "사용 방법을 묻거나 다른 댓글과 경험을 주고받는 반응은 어떻게 나타나는가?",
        "효과를 회의하거나 콘텐츠를 비판하는 댓글은 어떤 특징을 보이는가?",
        "영상이 댓글 작성자의 개인적 상황이나 경험과 연결되어 표현되는 양상은 무엇인가?",
    ]
    answers = [q1, q2, q3, q4, q5, q6]
    return [{"question": question, "answer": answer} for question, answer in zip(questions, answers)]


def build_conclusion(records: list[dict], stats: dict, report: dict, meta: dict) -> str:
    total = stats["total"]
    hope_n = _count(records, "hope")
    effect_n = _count(records, "effect")
    parts = []
    if meta.get("is_sample"):
        parts.append("이 결론은 기능 확인용 모의 데이터를 요약한 것이며, 실제 유튜브 댓글의 분석 결과가 아닙니다.")
    parts.append(
        f"대상 영상 ID는 {meta.get('video_id') or '-'}이고, 수집 시각은 {meta.get('collected_at') or '-'}입니다. "
        f"수집 {report.get('raw_count', 0)}건 가운데 중복 {report.get('duplicate_count', 0)}건과 "
        f"제외 {report.get('excluded_count', 0)}건을 뺀 분류 대상은 {total}건입니다. "
        "제외 기준은 홍보·스팸 의심, 무의미한 기호, 분석할 텍스트 없음입니다. "
        f"분류 방식은 {meta.get('classifier_name') or '규칙 기반'} ({meta.get('classifier_version') or '-'})이고, "
        f"판단 보류는 {stats.get('hold', 0)}건, 참고 점수 0.50 미만 또는 보류는 {stats.get('low_confidence', 0)}건입니다."
    )
    if total == 0:
        parts.append("분류 대상이 없어 이 자료만으로는 판단하기 어렵습니다.")
    else:
        parts.append(
            f"기대·희망형은 {hope_n}건({_percent(hope_n, total)}%), "
            f"효과를 경험했다고 서술한 댓글은 {effect_n}건({_percent(effect_n, total)}%)입니다. "
            "후자는 작성자가 그렇게 인식하거나 주장한 횟수이며, 주파수의 효능이 확인되었다는 뜻은 아닙니다."
        )
    parts.append(
        "댓글 작성자는 전체 시청자를 대표하지 않습니다. "
        "삭제·비공개 댓글과 API 수집 범위 밖의 댓글은 포함되지 않았고, "
        "청취와 이후 일의 선후가 댓글에 적혀 있어도 그것을 인과관계로 볼 수 없습니다."
    )
    if total and stats.get("hold", 0) / total >= 0.25:
        parts.append("판단 보류 비율이 높아 유형 분포를 단정하기 어렵습니다.")
    if total < 30 and not meta.get("is_sample"):
        parts.append("분류 대상이 적어 이 자료만으로는 판단하기 어렵습니다.")
    text = " ".join(parts)
    for forbidden in FORBIDDEN_CONCLUSION:
        if forbidden in text:
            raise RuntimeError("결론 문장 검사에 걸렸습니다.")
    return text


def summarize(records: list[dict], report: dict, meta: dict, title: str) -> dict:
    stats = category_stats(records)
    stopwords = load_stopwords()
    title_token_set = title_words(title)
    grouped_texts = {
        key: [record.get("text_analysis") or "" for record in records if record["label"] == key]
        for key in CATEGORY_ORDER
    }
    min_count = 2 if len(records) >= 12 else 1
    by_category = {
        key: keyword_table(texts, stopwords, title_token_set, min_count=1 if len(texts) < 8 else min_count, limit=8)
        for key, texts in grouped_texts.items()
    }
    overall = keyword_table(
        [record.get("text_analysis") or "" for record in records],
        stopwords,
        title_token_set,
        min_count=min_count,
        limit=15,
    )
    keywords = {
        "overall": overall,
        "by_category": by_category,
        "collocations": collocations(
            [record.get("text_analysis") or "" for record in records],
            stopwords,
        ),
        "title": title,
        "title_discount": "제목에 포함된 단어는 빈도에서 지우지 않고 반영 점수를 0.35배로 낮춥니다. 불용어 사전의 일반어는 제외합니다.",
    }
    replies = reply_overview(records)
    answers = research_answers(records, stats, keywords, replies, meta)
    conclusion = build_conclusion(records, stats, report, meta)
    return {
        "stats": stats,
        "keywords": keywords,
        "representatives": representatives(records),
        "replies": replies,
        "timeline": timeline(records),
        "answers": answers,
        "conclusion": conclusion,
        "comparison": comparison_rows(records, by_category, stats),
        "limitations": LIMITATIONS,
    }


def display_excerpt(record: dict, limit: int = 36) -> str:
    return excerpt(record.get("text_analysis") or "", limit)
