# -*- coding: utf-8 -*-
"""사용자에게 보여줄 오류. API 키와 추적 정보는 넣지 않는다."""

from __future__ import annotations


class UserFacingError(Exception):
    def __init__(self, title: str, detail: str, hint: str, code: str = "error"):
        self.title = title
        self.detail = detail
        self.hint = hint
        self.code = code
        super().__init__(title)

    def as_text(self) -> str:
        return f"{self.title}\n{self.detail}\n해결 방법: {self.hint}"
