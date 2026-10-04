# -*- coding: utf-8 -*-
"""선택 분류기.

기본 실행 경로에서는 만들지 않는다.
외부 LLM은 사용자가 전송에 동의하고 키를 넣은 경우에만 댓글 원문을 보낸다.
"""

from __future__ import annotations

import json
import os
import urllib.request

from src.categories import CATEGORY_ORDER, CATEGORIES
from src.classifier import Classification, CLASSIFIER_VERSION
from src.errors import UserFacingError

LLM_PROMPT = """당신은 한국어 유튜브 댓글을 아래 유형 중 하나로만 분류합니다.
effect, hope, usage, empathy, criticism, humor, hold
주 분류 하나와 보조 태그 배열, 0에서 1 사이의 참고 점수, 짧은 근거를 JSON으로 반환하세요.
주파수의 실제 효과를 판단하지 마세요.
댓글:
"""


class ExternalLLMClassifier:
    """OpenAI 호환 Chat Completions. 기본 설정에서는 사용하지 않는다."""

    name = "외부 LLM"
    version = "llm-optional-v1"
    sends_text_externally = True

    def __init__(self, api_key: str, base_url: str, model: str, confirmed: bool):
        if not confirmed:
            raise UserFacingError(
                "외부 전송에 동의하지 않았습니다",
                "LLM 분류는 댓글 원문을 외부 서비스로 보냅니다. 기본 분석은 규칙 기반이며 외부로 나가지 않습니다.",
                "규칙 기반 분류를 선택하거나, 전송 동의 확인란을 켠 뒤 다시 실행하세요.",
                code="llm_consent",
            )
        self.api_key = (api_key or os.environ.get("LLM_API_KEY") or "").strip()
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.model = (model or os.environ.get("LLM_MODEL") or "").strip()
        if not self.api_key or not self.model:
            raise UserFacingError(
                "LLM 설정이 없습니다",
                "외부 모델 이름 또는 LLM API 키를 찾지 못했습니다. 키 값은 표시하지 않습니다.",
                "LLM_API_KEY, LLM_MODEL 환경 변수를 설정하거나 화면 입력란을 채우세요. 없으면 규칙 기반 분류를 사용하세요.",
                code="llm_config",
            )

    def classify(self, text: str) -> Classification:
        payload = {
            "model": self.model,
            "temperature": 0,
            "messages": [
                {"role": "system", "content": "JSON만 반환합니다."},
                {"role": "user", "content": LLM_PROMPT + (text or "")},
            ],
        }
        request = urllib.request.Request(
            self.base_url + "/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=40) as response:
                body = json.loads(response.read().decode("utf-8"))
        except Exception:
            raise UserFacingError(
                "외부 LLM 호출에 실패했습니다",
                "댓글 분류 요청이 완료되지 않았습니다. 응답 본문과 키는 표시하지 않습니다.",
                "모델 이름, 엔드포인트, 네트워크를 확인하거나 규칙 기반 분류로 바꾸세요.",
                code="llm_call",
            ) from None
        try:
            content = body["choices"][0]["message"]["content"]
            parsed = json.loads(content)
            label = parsed.get("label")
            if label not in CATEGORY_ORDER:
                raise ValueError(label)
        except Exception:
            raise UserFacingError(
                "LLM 응답을 분류로 해석하지 못했습니다",
                "기대한 JSON 형식이 아니어서 이 댓글에 임의 유형을 넣지 않았습니다.",
                "규칙 기반 분류로 다시 실행하세요.",
                code="llm_parse",
            ) from None
        secondary = [item for item in parsed.get("secondary") or [] if item in CATEGORIES and item != label][:2]
        try:
            confidence = float(parsed.get("confidence") or 0.4)
        except (TypeError, ValueError):
            confidence = 0.4
        confidence = max(0.2, min(0.92, confidence))
        evidence = [str(parsed.get("reason") or "외부 모델이 제시한 근거"), "외부 LLM 분류이며 검증된 확률이 아닙니다."]
        return Classification(
            label=label,
            secondary=secondary,
            confidence=round(confidence, 2),
            evidence=evidence,
            hold_reason="" if label != "hold" else "모델이 판단 보류로 분류했습니다.",
            low_confidence=confidence < 0.5 or label == "hold",
            classifier_name=self.name,
            classifier_version=self.version,
        )


class JoblibModelClassifier:
    """학습된 모델 파일을 연결하는 자리. 파일이 없으면 동작하지 않는다."""

    name = "로컬 ML 모델"
    version = "ml-optional-v1"
    sends_text_externally = False

    def __init__(self, model_path: str):
        path = (model_path or os.environ.get("LLM_MODEL_PATH") or "").strip()
        if not path:
            raise UserFacingError(
                "학습된 모델 파일이 없습니다",
                "로컬 ML 분류를 선택했지만 모델 경로가 비어 있습니다. 검증되지 않은 정확도를 가정하지 않습니다.",
                "모델 파일 경로를 지정하거나 규칙 기반 분류를 사용하세요.",
                code="ml_missing",
            )
        try:
            import joblib
        except ImportError:
            raise UserFacingError(
                "모델 불러오기 도구가 없습니다",
                "joblib이 설치되어 있지 않아 모델 파일을 열 수 없습니다.",
                "규칙 기반 분류를 사용하세요. 로컬 모델이 필요할 때만 joblib을 설치하면 됩니다.",
                code="ml_import",
            ) from None
        try:
            self.model = joblib.load(path)
        except Exception:
            raise UserFacingError(
                "모델 파일을 열지 못했습니다",
                "경로의 파일을 분류 모델로 읽지 못했습니다. 경로 문자열은 화면에 그대로 반복하지 않습니다.",
                "파일 형식과 경로를 확인하거나 규칙 기반 분류를 사용하세요.",
                code="ml_load",
            ) from None
        self.version = f"ml-optional-v1 ({CLASSIFIER_VERSION} 인터페이스)"

    def classify(self, text: str) -> Classification:
        try:
            predicted = self.model.predict([text])[0]
        except Exception:
            raise UserFacingError(
                "로컬 모델 분류에 실패했습니다",
                "모델이 이 텍스트를 처리하지 못했습니다.",
                "규칙 기반 분류로 바꾸세요.",
                code="ml_predict",
            ) from None
        label = str(predicted)
        if label not in CATEGORIES:
            raise UserFacingError(
                "모델의 분류 라벨을 알 수 없습니다",
                "모델이 이 앱의 여섯 유형 또는 판단 보류 밖의 값을 반환했습니다.",
                "라벨이 effect, hope, usage, empathy, criticism, humor, hold 중 하나인 모델을 사용하세요.",
                code="ml_label",
            )
        return Classification(
            label=label,
            confidence=0.5,
            evidence=["로컬 모델 예측입니다. 별도 검증 수치를 제공하지 않아 참고 점수는 0.50으로 고정합니다."],
            low_confidence=True,
            hold_reason="" if label != "hold" else "모델이 판단 보류로 분류했습니다.",
            classifier_name=self.name,
            classifier_version=self.version,
        )
