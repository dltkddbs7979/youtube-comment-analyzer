# -*- coding: utf-8 -*-
"""설명 가능한 규칙 기반 분류기.

키워드 하나만으로 유형을 확정하지 않는다.
여러 규칙의 점수를 비교하고, 소망 어미가 있으면 경험 점수를 낮춘다.
신뢰도는 검증된 확률이 아니라 규칙 점수 차이로 만든 참고 점수다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re

from src.categories import CATEGORIES, CATEGORY_ORDER
from src.textutil import excerpt, has_emoji, meaningful_length

CLASSIFIER_NAME = "규칙 기반"
CLASSIFIER_VERSION = "rule-v1.0"

# README의 신뢰도 설명과 이 숫자를 맞춘다.
MIN_SCORE = 1.8
SECONDARY_MIN = 1.6
MARGIN_HOLD = 0.75
MARGIN_HOLD_TOP_BELOW = 3.4
THIN_LENGTH = 8
THIN_ALLOW_RULE = 2.2
LOW_CONFIDENCE = 0.50
CONFIDENCE_MAX = 0.92
THIN_CONFIDENCE_CAP = 0.60

CONFIDENCE_NOTE = (
    "신뢰도는 검증된 확률이 아닙니다. "
    "규칙 점수에서 1위와 2위의 차이에 0.08을 곱한 값(최대 0.28), "
    "가중치 1.6 이상인 규칙 하나당 0.05(최대 0.15), "
    "의미 있는 길이가 12자 이상이면 0.06, 20자 이상이면 0.12를 "
    "기본 0.42에 더합니다. 결과는 0.20 이상 0.92 이하로 자릅니다. "
    "의미 있는 길이가 8자 미만이면 0.60을 넘지 않게 제한합니다. "
    "판단 보류는 짧은 반응이면 0.22, 그 외에는 0.36입니다. "
    "0.50 미만은 검토 권장입니다."
)


@dataclass(frozen=True)
class Rule:
    category: str
    rule_id: str
    pattern: str
    weight: float
    description: str


RULES: list[Rule] = [
    Rule("effect", "contact_arrived", r"연락(?:이|은|을|도)?\s*(?:드디어\s*)?왔(?!으)", 2.6, "연락이 왔다는 과거·회고 표현"),
    Rule("effect", "contact_received", r"연락(?:을)?\s*받았(?!으)", 2.6, "연락을 받았다는 표현"),
    Rule("effect", "messenger", r"(?:카톡|문자)(?:이|을)?\s*(?:먼저\s*)?(?:왔|보냈|날아)(?!으)", 2.2, "메신저 연락이 왔다는 표현"),
    Rule("effect", "reunion", r"재회(?:를\s*)?(?:했|했어|했다|했음|했어요|했습니다|에\s*성공|됐|되었)", 2.6, "재회가 일어났다는 표현"),
    Rule("effect", "dating", r"사귀(?:게\s*)?(?:됐|되었|었|기\s*시작)", 2.4, "연애가 시작됐다는 표현"),
    Rule("effect", "confession", r"고백(?:을\s*)?받(?:았|아|음)(?!으)", 2.4, "고백을 받았다는 표현"),
    Rule("effect", "some_start", r"썸(?:을\s*)?타기\s*시작|썸(?:을\s*)?타게\s*(?:됐|되었)", 2.3, "썸이 시작됐다는 표현"),
    Rule("effect", "after_listening", r"듣고\s*나서|들은\s*지", 0.8, "청취 이후의 시간 순서"),
    Rule("effect", "felt_effect", r"효과(?:가|를)?\s*(?:좀\s*)?(?:있었|봤|보았|체감|느꼈)(?!으)", 1.6, "효과를 느꼈다는 표현"),
    Rule("hope", "wish_come", r"왔으면|오길|오게", 2.3, "오기를 바라는 소망 표현"),
    Rule("hope", "wish_become", r"됐으면|되었으면|있었으면|했으면|되길|되기를", 2.2, "이루어지기를 바라는 소망 표현"),
    Rule("hope", "wish_good", r"좋겠", 1.6, "좋기를 바라는 표현"),
    Rule("hope", "please", r"제발", 1.5, "간청 표현"),
    Rule("hope", "hope_verb", r"바라|바랍니다|바래요|희망|기도|기원|소원", 1.9, "바람·기원 표현"),
    Rule("hope", "want", r"싶(?:다|어|어요|습니다|음|네|군|져)", 1.3, "원하는 마음 표현"),
    Rule("hope", "come_true", r"이루어지|이뤄지|떠오르길", 1.6, "이루어지기를 바라는 표현"),
    Rule("hope", "commitment", r"듣고\s*자|들어\s*볼|믿어\s*보|틀어\s*놓", 1.4, "듣거나 믿어 보겠다는 다짐"),
    Rule("hope", "imperative_wish", r"와\s*주|와주|해주세요", 1.5, "일어나 달라는 부탁"),
    Rule("usage", "device", r"이어폰|헤드폰|스피커|에어팟", 1.8, "재생 기기를 묻는 표현"),
    Rule("usage", "volume", r"볼륨|음량", 1.8, "소리 크기를 묻는 표현"),
    Rule("usage", "count", r"몇\s*번|몇\s*회|몇\s*시간|몇\s*분|얼마나|며칠", 1.6, "횟수나 시간을 묻는 표현"),
    Rule("usage", "timing", r"자기\s*전|잘\s*때|자면서|출근", 1.4, "듣는 시점을 묻는 표현"),
    Rule("usage", "repeat", r"반복", 1.2, "반복 재생에 대한 표현"),
    Rule("usage", "listen_method", r"들어야|듣나요|들을까|재생", 1.4, "재생 방법에 대한 표현"),
    Rule("usage", "how", r"어떻게|방법", 1.0, "방법을 묻는 표현"),
    Rule("usage", "curious", r"궁금|알려\s*주", 1.2, "확인을 요청하는 표현"),
    Rule("usage", "question_ending", r"나요|까요|인가요|을까요|ㄹ까요", 0.8, "질문 어미"),
    Rule("usage", "question_mark", r"\?", 0.3, "물음표"),
    Rule("empathy", "agree", r"공감|동감", 1.8, "공감 표현"),
    Rule("empathy", "recommend", r"추천|공유|퍼갈", 1.6, "추천·공유 표현"),
    Rule("empathy", "thanks", r"감사(?:합니다|해요|요)|고맙", 1.8, "감사 표현"),
    Rule("empathy", "cheer", r"응원|화이팅|파이팅|힘내", 1.5, "응원 표현"),
    Rule("empathy", "good_video", r"좋은\s*영상|유익", 1.4, "영상 평가에 가까운 표현"),
    Rule("empathy", "me_too", r"저도(?:요|그래|같은|그렇게)|나도(?:요|그래)", 1.4, "같이 느낀다는 표현"),
    Rule("empathy", "comfort", r"힘이\s*(?:됩|돼|되)|위로", 1.4, "위로에 가까운 표현"),
    Rule("empathy", "save", r"저장", 0.8, "저장했다는 표현"),
    Rule("empathy", "friend", r"친구한테|친구에게", 1.2, "다른 사람에게 권하는 표현"),
    Rule("criticism", "scam", r"사기|가짜|허위|낚시", 2.3, "허위·사기라는 비판"),
    Rule("criticism", "placebo", r"플라시보|위약|확증\s*편향", 2.4, "플라시보 등 다른 설명"),
    Rule("criticism", "evidence", r"과학(?:적)?\s*근거(?:가)?\s*(?:없|부족|아니)|근거(?:가|를)?\s*(?:없|제시|부족)", 2.2, "근거를 문제 삼는 표현"),
    Rule("criticism", "superstition", r"미신|사이비", 2.0, "미신이라는 평가"),
    Rule("criticism", "no_effect", r"효과(?:가|는|도)?\s*(?:전혀\s*|하나도\s*)?(?:없|없었)", 2.2, "효과가 없었다는 평가"),
    Rule("criticism", "doubt", r"의심|믿을\s*수\s*없|반신반의", 1.8, "의심 표현"),
    Rule("criticism", "ad", r"광고", 1.5, "광고성을 문제 삼는 표현"),
    Rule("criticism", "deletion", r"삭제|조작", 1.2, "댓글 삭제·조작을 언급"),
    Rule("humor", "long_laugh", r"ㅋ{3,}|ㅎ{3,}", 1.7, "반복된 웃음 표현"),
    Rule("humor", "short_laugh", r"ㅋ{2}|ㅎ{2}", 0.6, "짧은 웃음 표현"),
    Rule("humor", "meme", r"레전드|실화냐|실화임|빵터|드립|밈|개웃", 1.7, "밈·과장 표현"),
    Rule("humor", "ridicule", r"무당이냐|귀신이냐|뭐야\s*이거", 1.8, "비꼼·조롱 표현"),
]

PERSONAL_TERMS = [
    (re.compile(r"짝사랑"), "짝사랑"),
    (re.compile(r"전남친|전여친|전\s*애인"), "전 연인"),
    (re.compile(r"헤어진|헤어졌|헤어지|이별"), "이별"),
    (re.compile(r"썸남|썸녀|썸을|썸(?=[이가은는을를\s])"), "썸"),
    (re.compile(r"읽씹"), "읽씹"),
    (re.compile(r"잠수"), "잠수"),
    (re.compile(r"남자친구|여자친구|남친|여친"), "연인"),
    (re.compile(r"남편|아내"), "배우자"),
    (re.compile(r"좋아하는\s*사람"), "좋아하는 사람"),
]

_COMPILED = [(rule, re.compile(rule.pattern, re.IGNORECASE)) for rule in RULES]


@dataclass
class Classification:
    label: str
    secondary: list[str] = field(default_factory=list)
    confidence: float = 0.0
    evidence: list[str] = field(default_factory=list)
    hold_reason: str = ""
    scores: dict[str, float] = field(default_factory=dict)
    low_confidence: bool = True
    personal_terms: list[str] = field(default_factory=list)
    classifier_name: str = CLASSIFIER_NAME
    classifier_version: str = CLASSIFIER_VERSION

    def to_fields(self) -> dict:
        return {
            "label": self.label,
            "label_name": CATEGORIES[self.label]["name"],
            "original_label": self.label,
            "original_label_name": CATEGORIES[self.label]["name"],
            "secondary": list(self.secondary),
            "secondary_names": [CATEGORIES[item]["name"] for item in self.secondary],
            "confidence": self.confidence,
            "evidence": list(self.evidence),
            "auto_evidence": list(self.evidence),
            "hold_reason": self.hold_reason,
            "scores": {key: round(value, 2) for key, value in self.scores.items()},
            "low_confidence": self.low_confidence,
            "personal_terms": list(self.personal_terms),
            "personal_context": bool(self.personal_terms),
            "classifier_name": self.classifier_name,
            "classifier_version": self.classifier_version,
            "user_modified": False,
        }


def _negated(text: str, start: int, end: int) -> bool:
    span = text[start:end]
    before = text[max(0, end - 8) : end]
    if re.search(r"(안|못)\s*[가-힣]{0,4}$", span):
        return True
    if re.search(r"(안|못)\s*$", before[: max(0, len(before) - 1)]):
        return True
    return False


def _laughter_chars(text: str) -> int:
    return len(re.findall(r"[ㅋㅎ]", text))


def _personal_terms(text: str) -> list[str]:
    found: list[str] = []
    for pattern, name in PERSONAL_TERMS:
        if pattern.search(text) and name not in found:
            found.append(name)
    return found


def _confidence(top: float, second: float, strong_rules: int, length: int, thin: bool, hold: bool, only_reaction: bool) -> float:
    if hold:
        return 0.22 if only_reaction else 0.36
    margin = max(0.0, top - second)
    score = 0.42 + min(0.28, margin * 0.08) + min(0.15, strong_rules * 0.05)
    if length >= 20:
        score += 0.12
    elif length >= 12:
        score += 0.06
    score = max(0.20, min(CONFIDENCE_MAX, score))
    if thin:
        score = min(score, THIN_CONFIDENCE_CAP)
    return round(score, 2)


def classify_text(text: str) -> Classification:
    raw = text or ""
    scores = {key: 0.0 for key in CATEGORY_ORDER if key != "hold"}
    evidence_by_cat: dict[str, list[str]] = {key: [] for key in scores}
    strongest = {key: 0.0 for key in scores}
    strong_rule_count = 0

    for rule, pattern in _COMPILED:
        match = pattern.search(raw)
        if match is None:
            continue
        if rule.category == "effect" and _negated(raw, match.start(), match.end()):
            continue
        scores[rule.category] += rule.weight
        strongest[rule.category] = max(strongest[rule.category], rule.weight)
        if rule.weight >= 1.6:
            strong_rule_count += 1
        snippet = excerpt(match.group(0), 40)
        evidence_by_cat[rule.category].append(f"{rule.description} (일치: '{snippet}')")

    laughter = _laughter_chars(raw)
    if laughter >= 3 and re.search(r"아니라|실화|레전드|무당|귀신", raw):
        scores["humor"] += 1.2
        evidence_by_cat["humor"].append("웃음과 비꼼이 함께 있어 조롱 점수를 더함")

    if scores["hope"] >= 2.0:
        scores["effect"] *= 0.25
    if laughter >= 3 and scores["humor"] >= 2.0:
        scores["effect"] *= 0.45
        evidence_by_cat["humor"].append("웃음·조롱이 중심이라 결과 서술을 경험으로 그대로 받지 않음")

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    top_label, top_score = ranked[0]
    second_label, second_score = ranked[1]
    length = meaningful_length(raw)
    thin = length < THIN_LENGTH
    only_reaction = length < 2 or (length == 0 and has_emoji(raw))

    hold_reason = ""
    label = top_label
    if only_reaction:
        label = "hold"
        hold_reason = "짧은 감탄, 이모지 또는 웃음 기호만 있어 의도를 정하기 어렵습니다."
    elif top_score < MIN_SCORE:
        label = "hold"
        hold_reason = "유형을 정할 규칙 점수가 약합니다."
    elif thin and strongest[top_label] < THIN_ALLOW_RULE:
        label = "hold"
        hold_reason = "텍스트가 짧아 문맥이 부족합니다."
    elif (top_score - second_score) < MARGIN_HOLD and second_score >= 1.5 and top_score < MARGIN_HOLD_TOP_BELOW:
        label = "hold"
        hold_reason = (
            f"상위 유형의 점수가 비슷합니다. "
            f"1위 {CATEGORIES[top_label]['name']} {top_score:.2f}점, "
            f"2위 {CATEGORIES[second_label]['name']} {second_score:.2f}점."
        )

    secondary: list[str] = []
    if label != "hold":
        for category, score in ranked:
            if category == label:
                continue
            if score >= SECONDARY_MIN and score >= top_score * 0.35:
                secondary.append(category)
            if len(secondary) >= 2:
                break

    visible = [(key, value) for key, value in ranked if value > 0]
    if visible:
        score_line = "규칙 점수: " + ", ".join(
            f"{CATEGORIES[key]['name']} {value:.2f}" for key, value in visible
        )
    else:
        score_line = "규칙 점수: 적용된 규칙 없음"
    score_line += " (참고 점수이며 확률이 아님)"

    if label == "hold":
        evidence = [f"판단 보류: {hold_reason}", score_line]
    else:
        evidence = evidence_by_cat[label][:4] + [score_line]
        if secondary:
            names = ", ".join(CATEGORIES[item]["name"] for item in secondary)
            evidence.append(f"보조 태그: {names}")

    confidence = _confidence(
        top_score,
        second_score,
        strong_rule_count,
        length,
        thin,
        label == "hold",
        only_reaction,
    )
    return Classification(
        label=label,
        secondary=secondary,
        confidence=confidence,
        evidence=evidence,
        hold_reason=hold_reason,
        scores=scores,
        low_confidence=confidence < LOW_CONFIDENCE or label == "hold",
        personal_terms=_personal_terms(raw),
    )


class RuleClassifier:
    """기본 분류기. 외부로 댓글을 보내지 않는다."""

    name = CLASSIFIER_NAME
    version = CLASSIFIER_VERSION

    def classify(self, text: str) -> Classification:
        return classify_text(text)


def apply_user_label(record: dict, new_label_id: str) -> None:
    """사용자가 고친 주 분류를 기록한다. 참고 점수는 자동 분류 값 그대로 둔다."""
    record["label"] = new_label_id
    record["label_name"] = CATEGORIES[new_label_id]["name"]
    record["user_modified"] = new_label_id != record.get("original_label")
    auto = list(record.get("auto_evidence") or [])
    if record["user_modified"]:
        record["evidence"] = [
            "사용자가 주 분류를 수정했습니다. 신뢰도는 수정 전 자동 분류에 대한 참고 점수이며 확률이 아닙니다."
        ] + auto
    else:
        record["evidence"] = auto
    if new_label_id != "hold":
        record["hold_reason"] = ""
