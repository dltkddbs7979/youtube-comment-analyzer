# -*- coding: utf-8 -*-
"""YouTube Data API v3 댓글 수집.

비공식 스크래핑은 쓰지 않는다.
API 키는 예외 문구, 반환 데이터, 로그에 넣지 않는다.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
import time

import requests

from src.errors import UserFacingError

API_ROOT = "https://www.googleapis.com/youtube/v3"
MAX_PAGES = 80
MAX_REPLY_PAGES_PER_THREAD = 10
RETRY_LIMIT = 3
RETRY_WAITS = (0.8, 1.6, 3.2)
DEFAULT_COMMENT_CAP = 5000
HARD_COMMENT_CAP = 20000
REQUEST_TIMEOUT = 20

VIDEO_ID_RE = re.compile(r"^[A-Za-z0-9_-]{11}$")
URL_PATTERNS = [
    re.compile(r"(?:v=|/embed/|/shorts/|youtu\.be/|/live/)([A-Za-z0-9_-]{11})"),
]


@dataclass
class CollectionResult:
    comments: list[dict] = field(default_factory=list)
    pages: int = 0
    reply_pages: int = 0
    video_title: str = ""
    api_comment_count: int | None = None
    warning: str = ""
    partial: bool = False


def parse_video_id(value: str) -> str:
    text = (value or "").strip()
    if VIDEO_ID_RE.fullmatch(text):
        return text
    for pattern in URL_PATTERNS:
        found = pattern.search(text)
        if found:
            return found.group(1)
    raise UserFacingError(
        "유튜브 주소 또는 영상 ID가 올바르지 않습니다",
        "영상 ID 11자리, watch 주소, youtu.be 주소, shorts 주소를 넣을 수 있습니다.",
        "예: https://www.youtube.com/watch?v=15Gbo7Xcy80",
        code="bad_video_id",
    )


def validate_api_key_format(api_key: str) -> str:
    key = (api_key or "").strip()
    if not key:
        raise UserFacingError(
            "API 키가 없습니다",
            "환경 변수 YOUTUBE_API_KEY와 화면 입력란 모두에서 키를 찾지 못했습니다.",
            "Google Cloud에서 YouTube Data API v3를 켜고 API 키를 만든 뒤, .env 파일이나 화면의 입력란에 넣으세요. "
            "화면 입력 키는 이 세션 메모리에서만 쓰고 파일로 저장하지 않습니다. "
            "키가 없으면 모의 데이터로 화면만 먼저 확인할 수 있습니다.",
            code="missing_key",
        )
    if any(char.isspace() for char in key) or len(key) < 20 or len(key) > 256:
        raise UserFacingError(
            "API 키 형식이 올바르지 않습니다",
            "키에 공백이 있거나 길이가 예상 범위 밖입니다. 키 값은 표시하지 않습니다.",
            "Google Cloud 사용자 인증 정보에서 키를 다시 복사해 공백 없이 넣으세요. .env.example 형식을 확인하세요.",
            code="bad_key_format",
        )
    return key


def map_api_error(status: int, payload: dict | None) -> UserFacingError:
    """HTTP 상태와 reason만 보고 한국어 안내로 바꾼다. 원문 메시지는 화면에 올리지 않는다."""
    reason = ""
    if isinstance(payload, dict):
        error = payload.get("error") or {}
        errors = error.get("errors") or []
        if errors and isinstance(errors[0], dict):
            reason = str(errors[0].get("reason") or "")
        status_text = str(error.get("status") or "")
    else:
        status_text = ""
    reason_l = reason.lower()
    status_l = status_text.lower()

    if status == 404 or "videonotfound" in reason_l:
        return UserFacingError(
            "영상을 찾을 수 없습니다",
            "해당 영상 ID로 공개 영상을 확인하지 못했습니다.",
            "주소와 영상 ID를 다시 확인하세요. 비공개이거나 삭제된 영상일 수 있습니다.",
            code="video_not_found",
        )
    if "commentsdisabled" in reason_l:
        return UserFacingError(
            "댓글이 비활성화되어 있습니다",
            "이 영상은 댓글을 받지 않도록 설정되어 있어 수집할 공개 댓글이 없습니다.",
            "다른 영상을 선택하거나, 댓글이 켜진 영상인지 확인하세요.",
            code="comments_disabled",
        )
    if "quotaexceeded" in reason_l or "dailylimitexceeded" in reason_l:
        return UserFacingError(
            "YouTube API 할당량을 초과했습니다",
            "오늘 사용할 수 있는 호출 한도를 넘겼습니다. 이미 받은 댓글이 있으면 그 범위만 분석합니다.",
            "Google Cloud 할당량이 초기화된 뒤 다시 실행하세요. 최대 수집 수를 낮추면 호출 수를 줄일 수 있습니다. "
            "할당량 제한을 우회하는 수집은 지원하지 않습니다.",
            code="quota",
        )
    if status == 429 or "ratelimit" in reason_l:
        return UserFacingError(
            "요청이 일시적으로 제한되었습니다",
            "짧은 시간에 호출이 많아 YouTube가 응답을 거절했습니다.",
            "잠시 후 다시 실행하세요. 재시도는 3회로 제한되어 있습니다.",
            code="rate_limit",
        )
    if status in {500, 503} or "backenderror" in reason_l:
        return UserFacingError(
            "YouTube API가 일시적으로 응답하지 않습니다",
            "서버 오류가 반복되었습니다.",
            "잠시 후 다시 실행하세요. 네트워크 상태를 함께 확인하세요.",
            code="temporary",
        )
    if "keyinvalid" in reason_l or "api_key_invalid" in reason_l or status_l == "invalid_argument" or (
        status == 400 and "badrequest" in reason_l
    ):
        return UserFacingError(
            "API 키가 유효하지 않거나 요청이 거절되었습니다",
            "키 권한, API 사용 설정, 또는 영상 ID를 확인해 주세요. 키 값은 표시하지 않습니다.",
            "YouTube Data API v3가 사용 설정된 프로젝트의 키인지 확인하세요. 키를 화면에 붙여 넣은 경우 앞뒤 공백이 없는지도 보세요.",
            code="invalid_key",
        )
    if status in {401, 403} or "forbidden" in reason_l or "insufficient" in reason_l:
        return UserFacingError(
            "API 접근 권한이 없습니다",
            "키로는 이 영상의 댓글을 가져올 수 없습니다. HTTP 상태 코드만 확인했습니다.",
            "키가 YouTube Data API v3를 호출할 수 있는지, 영상 댓글이 공개인지 확인하세요.",
            code="forbidden",
        )
    if status == 400:
        return UserFacingError(
            "요청 형식이 올바르지 않습니다",
            "영상 ID나 요청 매개변수를 확인하지 못했습니다.",
            "watch 주소 전체를 다시 붙여 넣거나 영상 ID 11자리만 입력하세요.",
            code="bad_request",
        )
    return UserFacingError(
        "댓글을 가져오지 못했습니다",
        f"YouTube API가 HTTP {status} 상태로 응답했습니다. 응답 본문은 표시하지 않습니다.",
        "네트워크, API 키, 할당량, 영상의 댓글 설정을 순서대로 확인하세요.",
        code="api_error",
    )


class YouTubeClient:
    def __init__(self, api_key: str, session: requests.Session | None = None):
        self.api_key = validate_api_key_format(api_key)
        self.session = session or requests.Session()

    def _get(self, path: str, params: dict) -> dict:
        url = f"{API_ROOT}/{path}"
        last_error: UserFacingError | None = None
        for attempt in range(RETRY_LIMIT):
            try:
                response = self.session.get(
                    url,
                    params={**params, "key": self.api_key},
                    timeout=REQUEST_TIMEOUT,
                )
            except requests.Timeout:
                last_error = UserFacingError(
                    "요청 시간이 초과되었습니다",
                    "YouTube API 응답이 제한 시간 안에 도착하지 않았습니다.",
                    "네트워크 연결을 확인한 뒤 다시 실행하세요.",
                    code="timeout",
                )
                time.sleep(RETRY_WAITS[attempt])
                continue
            except requests.RequestException:
                last_error = UserFacingError(
                    "네트워크 오류로 API에 연결하지 못했습니다",
                    "인터넷 연결이 불안정하거나 접속이 막혀 있습니다.",
                    "연결을 확인한 뒤 다시 실행하세요. 프록시를 쓴다면 해당 설정도 확인하세요.",
                    code="network",
                )
                time.sleep(RETRY_WAITS[attempt])
                continue

            if response.status_code == 200:
                try:
                    return response.json()
                except ValueError:
                    raise UserFacingError(
                        "API 응답을 해석하지 못했습니다",
                        "JSON이 아닌 응답이 도착했습니다.",
                        "잠시 후 다시 실행하세요.",
                        code="bad_response",
                    ) from None

            payload = None
            try:
                payload = response.json()
            except ValueError:
                payload = None
            mapped = map_api_error(response.status_code, payload)
            if mapped.code in {"temporary", "rate_limit", "timeout"} and attempt < RETRY_LIMIT - 1:
                last_error = mapped
                time.sleep(RETRY_WAITS[attempt])
                continue
            raise mapped
        assert last_error is not None
        raise last_error

    def fetch_video(self, video_id: str) -> tuple[str, int | None]:
        data = self._get(
            "videos",
            {"part": "snippet,statistics", "id": video_id, "maxResults": 1},
        )
        items = data.get("items") or []
        if not items:
            raise UserFacingError(
                "영상을 찾을 수 없습니다",
                "API가 해당 영상 정보를 반환하지 않았습니다.",
                "영상 ID가 맞는지, 영상이 공개 상태인지 확인하세요.",
                code="video_not_found",
            )
        snippet = items[0].get("snippet") or {}
        statistics = items[0].get("statistics") or {}
        title = str(snippet.get("title") or "")
        count = statistics.get("commentCount")
        try:
            comment_count = int(count) if count is not None else None
        except (TypeError, ValueError):
            comment_count = None
        return title, comment_count

    def collect(
        self,
        video_id: str,
        max_comments: int = DEFAULT_COMMENT_CAP,
        include_replies: bool = True,
        progress=None,
    ) -> CollectionResult:
        video_id = parse_video_id(video_id)
        cap = DEFAULT_COMMENT_CAP if not max_comments else min(int(max_comments), HARD_COMMENT_CAP)
        title, api_count = self.fetch_video(video_id)
        result = CollectionResult(video_title=title, api_comment_count=api_count)
        seen: set[str] = set()
        page_token = None

        while result.pages < MAX_PAGES and len(result.comments) < cap:
            params = {
                "part": "snippet,replies",
                "videoId": video_id,
                "maxResults": 100,
                "textFormat": "plainText",
                "order": "time",
            }
            if page_token:
                params["pageToken"] = page_token
            try:
                data = self._get("commentThreads", params)
            except UserFacingError as exc:
                if exc.code == "quota" and result.comments:
                    result.warning = exc.detail
                    result.partial = True
                    break
                raise
            result.pages += 1
            for item in data.get("items") or []:
                if len(result.comments) >= cap:
                    result.warning = f"설정한 최대 수집 수 {cap}건에 도달해 이후 댓글은 가져오지 않았습니다."
                    result.partial = True
                    break
                snippet = item.get("snippet") or {}
                top = snippet.get("topLevelComment") or {}
                parsed = _parse_comment(top, video_id, is_reply=False, parent_id=None)
                _append(result.comments, seen, parsed)
                included = (item.get("replies") or {}).get("comments") or []
                for reply in included:
                    if len(result.comments) >= cap:
                        break
                    reply_parsed = _parse_comment(
                        reply,
                        video_id,
                        is_reply=True,
                        parent_id=parsed["comment_id"],
                    )
                    _append(result.comments, seen, reply_parsed)
                total_replies = int(snippet.get("totalReplyCount") or 0)
                if include_replies and total_replies > len(included) and len(result.comments) < cap:
                    extra_pages, warning = self._collect_rest_replies(
                        parsed["comment_id"],
                        video_id,
                        seen,
                        result.comments,
                        cap,
                    )
                    result.reply_pages += extra_pages
                    if warning:
                        result.warning = warning
                        result.partial = True
                        break
            if progress:
                progress(
                    {
                        "pages": result.pages,
                        "comments": len(result.comments),
                        "replies": sum(1 for item in result.comments if item["is_reply"]),
                        "message": "댓글 스레드를 수집하는 중",
                    }
                )
            if result.partial:
                break
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        else:
            if result.pages >= MAX_PAGES and page_token:
                result.warning = "페이지 상한에 도달했습니다. 일부 댓글만 수집되었을 수 있습니다."
                result.partial = True

        if not result.comments and not result.warning:
            result.warning = "공개 댓글이 없습니다. 댓글이 없거나, 검토 대기·스팸 필터로 API에 나오지 않았을 수 있습니다."
        if api_count is not None and len(result.comments) < api_count:
            gap = (
                f"API가 알려 준 댓글 수는 {api_count}건이고 이번에 수집한 수는 {len(result.comments)}건입니다. "
                "삭제, 비공개, 검토 대기, 대댓글 누락, 최대 수집 수 때문에 차이가 날 수 있습니다."
            )
            result.warning = f"{result.warning} {gap}".strip()
        return result

    def _collect_rest_replies(self, parent_id: str, video_id: str, seen: set[str], bucket: list[dict], cap: int):
        pages = 0
        page_token = None
        warning = ""
        while pages < MAX_REPLY_PAGES_PER_THREAD and len(bucket) < cap:
            params = {
                "part": "snippet",
                "parentId": parent_id,
                "maxResults": 100,
                "textFormat": "plainText",
            }
            if page_token:
                params["pageToken"] = page_token
            try:
                data = self._get("comments", params)
            except UserFacingError as exc:
                if exc.code in {"quota", "rate_limit", "temporary"}:
                    return pages, exc.detail
                raise
            pages += 1
            for item in data.get("items") or []:
                if len(bucket) >= cap:
                    warning = f"설정한 최대 수집 수 {cap}건에 도달해 이후 대댓글은 가져오지 않았습니다."
                    return pages, warning
                parsed = _parse_comment(item, video_id, is_reply=True, parent_id=parent_id)
                _append(bucket, seen, parsed)
            page_token = data.get("nextPageToken")
            if not page_token:
                break
        if page_token:
            warning = "대댓글 페이지 상한에 도달해 일부 대댓글이 빠졌을 수 있습니다."
        return pages, warning


def _append(bucket: list[dict], seen: set[str], comment: dict) -> None:
    comment_id = comment.get("comment_id") or ""
    if not comment_id or comment_id in seen:
        return
    seen.add(comment_id)
    bucket.append(comment)


def _parse_comment(item: dict, video_id: str, is_reply: bool, parent_id: str | None) -> dict:
    snippet = item.get("snippet") or {}
    text = snippet.get("textOriginal")
    if text is None:
        text = snippet.get("textDisplay") or ""
    parent = parent_id or snippet.get("parentId") or None
    return {
        "comment_id": str(item.get("id") or ""),
        "parent_id": parent if is_reply else None,
        "video_id": video_id,
        "text": str(text),
        "published_at": str(snippet.get("publishedAt") or ""),
        "like_count": int(snippet.get("likeCount") or 0),
        "is_reply": is_reply or bool(snippet.get("parentId")),
    }


def fetch_oembed_title(video_id: str) -> str:
    """댓글이 아닌 공개 제목만 가져온다. 실패해도 분석을 막지 않는다."""
    url = "https://www.youtube.com/oembed"
    try:
        response = requests.get(
            url,
            params={"url": f"https://www.youtube.com/watch?v={video_id}", "format": "json"},
            timeout=5,
        )
        if response.status_code != 200:
            return ""
        title = response.json().get("title") or ""
        return str(title)
    except (requests.RequestException, ValueError):
        return ""
