# -*- coding: utf-8 -*-
"""댓글 분류 체계.

주 분류는 아래 여섯 유형 또는 판단 보류 중 하나다.
한 댓글이 여러 유형에 걸치면 규칙 점수가 가장 높은 유형을 주 분류로 두고,
일정 점수 이상의 다른 유형은 보조 태그로 남긴다.
"""

from __future__ import annotations

CATEGORY_ORDER = [
    "effect",
    "hope",
    "usage",
    "empathy",
    "criticism",
    "humor",
    "hold",
]

CATEGORIES = {
    "effect": {
        "id": "effect",
        "name": "효과 경험형",
        "symbol": "●",
        "color": "#0072B2",
        "hatch": "",
        "summary": "들은 뒤 원하던 일이 일어났다고 본인 경험을 서술",
        "criteria": (
            "작성자 자신이 주파수를 들은 뒤 연락, 재회, 고백, 연애 시작처럼 "
            "원하던 일이 일어났다고 말할 때 해당한다. "
            "'왔으면', '되길'처럼 아직 일어나지 않은 소망이면 효과 경험으로 보지 않는다. "
            "이 유형은 작성자가 효과를 경험했다고 인식하거나 주장한 댓글이지, "
            "주파수의 실제 효과나 인과관계의 증거가 아니다."
        ),
    },
    "hope": {
        "id": "hope",
        "name": "기대·희망형",
        "symbol": "▲",
        "color": "#E69F00",
        "hatch": "//",
        "summary": "원하는 일이 이루어지기를 바라거나 다짐·기원",
        "criteria": (
            "연락, 재회, 연애처럼 원하는 일이 일어나기를 바라거나 "
            "들겠다는 다짐, 기원, 희망을 말할 때 해당한다. "
            "'연락 왔으면 좋겠다'는 경험이 아니라 기대·희망이다."
        ),
    },
    "usage": {
        "id": "usage",
        "name": "사용 방법 질문형",
        "symbol": "■",
        "color": "#009E73",
        "hatch": "..",
        "summary": "재생 시간, 횟수, 기기, 볼륨, 듣는 시점을 물음",
        "criteria": (
            "재생 시간, 듣는 횟수, 이어폰·스피커 여부, 볼륨, 듣는 시점처럼 "
            "이용 방법을 묻거나 확인할 때 해당한다. "
            "방법 질문 안에 소망이 함께 있고 질문 행위가 더 분명하면 "
            "주 분류는 사용 방법 질문형, 소망은 보조 태그로 둔다."
        ),
    },
    "empathy": {
        "id": "empathy",
        "name": "공감·추천형",
        "symbol": "◆",
        "color": "#CC79A7",
        "hatch": "xx",
        "summary": "공감, 추천·공유, 감사, 응원이 중심",
        "criteria": (
            "다른 사람의 말에 공감하거나 영상을 추천·공유하고, "
            "감사나 응원을 중심으로 말할 때 해당한다. "
            "본인이 결과를 겪었다고 말하는 내용이 더 분명하면 효과 경험형을 우선한다."
        ),
    },
    "criticism": {
        "id": "criticism",
        "name": "회의·비판형",
        "symbol": "×",
        "color": "#D55E00",
        "hatch": "++",
        "summary": "효과가 의심되거나 근거·콘텐츠를 비판",
        "criteria": (
            "효과가 없다고 말하거나, 과학적 근거, 광고성, 미신, 사기 여부를 "
            "문제 삼을 때 해당한다. "
            "농담과 비꼼이 글의 중심이면 유머·조롱형을 우선하고 비판은 보조 태그로 둘 수 있다."
        ),
    },
    "humor": {
        "id": "humor",
        "name": "유머·조롱형",
        "symbol": "★",
        "color": "#56B4E9",
        "hatch": "oo",
        "summary": "농담, 밈, 풍자, 비꼼이 중심",
        "criteria": (
            "농담, 밈, 풍자, 비꼼, 조롱이 글의 중심일 때 해당한다. "
            "웃음 기호만 있고 다른 내용이 없으면 유머로 확정하지 않고 판단 보류로 둔다. "
            "웃음과 조롱이 중심이면, 농담으로 적은 결과 서술을 효과 경험으로 그대로 받지 않는다."
        ),
    },
    "hold": {
        "id": "hold",
        "name": "판단 보류",
        "symbol": "○",
        "color": "#5C6770",
        "hatch": "--",
        "summary": "문맥 부족, 짧은 반응, 유형 점수가 비슷함",
        "criteria": (
            "짧은 감탄, 이모지만 있는 글, 문맥을 알 수 없는 글, "
            "규칙 점수가 약하거나 상위 유형의 점수가 비슷해 "
            "핵심 의도를 하나로 정하기 어려운 글이다. "
            "별도 항목으로 집계하며 비율의 분모에는 포함된다."
        ),
    },
}


def ordered_categories():
    return [CATEGORIES[key] for key in CATEGORY_ORDER]


def name_to_id(name: str) -> str:
    for key, item in CATEGORIES.items():
        if item["name"] == name:
            return key
    raise KeyError(name)


def label_text(category_id: str) -> str:
    item = CATEGORIES[category_id]
    return f"{item['symbol']} {item['name']}"
