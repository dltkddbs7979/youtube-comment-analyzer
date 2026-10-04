# -*- coding: utf-8 -*-
"""한국어 Streamlit 화면."""

from __future__ import annotations

import html
import os
from pathlib import Path

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from src.categories import CATEGORIES, name_to_id, ordered_categories
from src.charts import category_chart_html, font_caption, keyword_chart_html, render_category_chart
from src.classifier import CONFIDENCE_NOTE, RuleClassifier, apply_user_label
from src.errors import UserFacingError
from src.export_data import analysis_csv_bytes, analysis_json_bytes, excluded_csv_bytes, file_stamp
from src.fonts_setup import font_status, ui_font_css
from src.keywords import analyzer_status
from src.optional_models import ExternalLLMClassifier, JoblibModelClassifier
from src.pipeline import DEFAULT_VIDEO_ID, DEFAULT_VIDEO_URL, KNOWN_TITLES, now_iso, refresh_summary, run_prepared
from src.sample_data import load_sample_comments
from src.summarize import REPRESENTATIVE_LIMIT, display_excerpt
from src.youtube_api import YouTubeClient, fetch_oembed_title, parse_video_id

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

CATEGORY_NAMES = [item["name"] for item in ordered_categories()]


def inject_style() -> None:
    st.markdown(
        f"""
        <style>
        {ui_font_css()}
        .block-container {{
            max-width: 1080px;
            padding-top: 1.2rem;
            padding-bottom: 3rem;
        }}
        h1 {{ font-size: 1.85rem; letter-spacing: -0.03em; }}
        h2 {{ font-size: 1.35rem; letter-spacing: -0.02em; }}
        p, li, label {{ font-size: 1.02rem; line-height: 1.65; }}
        .comment-card {{
            border: 1px solid #D5DDE6;
            border-left: 6px solid #0F5C8C;
            border-radius: 10px;
            padding: 0.85rem 1rem 0.7rem;
            margin: 0.4rem 0 0.8rem;
            background: #ffffff;
            word-break: break-word;
            overflow-wrap: anywhere;
        }}
        .comment-card .body {{
            white-space: pre-wrap;
            line-height: 1.7;
            font-size: 1.02rem;
            color: #142033;
        }}
        .comment-card .meta {{
            color: #3E4A59;
            font-size: 0.95rem;
            margin-bottom: 0.35rem;
        }}
        [data-testid="stMetricValue"] {{ font-size: 1.6rem; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def show_error(error: UserFacingError) -> None:
    st.error(f"**{error.title}**\n\n{error.detail}\n\n해결 방법: {error.hint}")


def api_key_from_state() -> str:
    typed = str(st.session_state.get("api_key_input") or "").strip()
    if typed:
        return typed
    return os.environ.get("YOUTUBE_API_KEY", "").strip()


def key_status_text() -> str:
    typed = bool(str(st.session_state.get("api_key_input") or "").strip())
    env_set = bool(os.environ.get("YOUTUBE_API_KEY", "").strip())
    if typed and env_set:
        return "세션에 입력한 키를 우선 사용합니다. 환경 변수에도 키가 있습니다. 키 값은 표시하지 않습니다."
    if typed:
        return "세션에 입력한 키를 사용합니다. 파일로 저장하지 않으며 화면에도 값을 표시하지 않습니다."
    if env_set:
        return "환경 변수 YOUTUBE_API_KEY가 설정되어 있습니다. 키 값은 표시하지 않습니다."
    return "사용 가능한 API 키가 없습니다. 키를 설정하거나 모의 데이터로 화면을 확인하세요."


def build_classifier():
    kind = st.session_state.get("classifier_kind") or "규칙 기반 (외부 전송 없음)"
    if kind.startswith("규칙"):
        return RuleClassifier()
    if kind.startswith("외부"):
        return ExternalLLMClassifier(
            api_key=str(st.session_state.get("llm_key_input") or ""),
            base_url=str(st.session_state.get("llm_base") or ""),
            model=str(st.session_state.get("llm_model") or ""),
            confirmed=bool(st.session_state.get("llm_consent")),
        )
    return JoblibModelClassifier(str(st.session_state.get("ml_path") or ""))


def run_sample() -> dict:
    comments, notice, title = load_sample_comments()
    meta = {
        "video_id": DEFAULT_VIDEO_ID,
        "video_url": DEFAULT_VIDEO_URL,
        "video_title": title or KNOWN_TITLES.get(DEFAULT_VIDEO_ID, ""),
        "source": "sample",
        "is_sample": True,
        "notice": notice,
        "pages": 1,
        "reply_pages": 0,
        "api_comment_count": None,
        "partial": False,
        "warning": "모의 데이터라서 YouTube API를 호출하지 않았습니다.",
        "collected_at": now_iso(),
    }
    return run_prepared(comments, meta, build_classifier())


def run_api(url: str, max_comments: int, include_replies: bool) -> dict:
    video_id = parse_video_id(url)
    client = YouTubeClient(api_key_from_state())
    progress_box = st.empty()

    def progress(event: dict) -> None:
        progress_box.info(
            f"수집 중 · 페이지 {event['pages']} · 댓글 {event['comments']}건 · 대댓글 {event['replies']}건. "
            "전체 페이지 수는 응답이 끝나야 알 수 있습니다."
        )

    collected = client.collect(
        video_id,
        max_comments=max_comments,
        include_replies=include_replies,
        progress=progress,
    )
    progress_box.success(
        f"수집을 마쳤습니다. 페이지 {collected.pages} · 댓글 {len(collected.comments)}건 · "
        f"대댓글 추가 요청 {collected.reply_pages}회"
    )
    title = collected.video_title or fetch_oembed_title(video_id) or KNOWN_TITLES.get(video_id, "")
    meta = {
        "video_id": video_id,
        "video_url": f"https://www.youtube.com/watch?v={video_id}",
        "video_title": title,
        "source": "youtube_api",
        "is_sample": False,
        "notice": "",
        "pages": collected.pages,
        "reply_pages": collected.reply_pages,
        "api_comment_count": collected.api_comment_count,
        "partial": collected.partial,
        "warning": collected.warning,
        "collected_at": now_iso(),
    }
    return run_prepared(collected.comments, meta, build_classifier())


def render_settings() -> None:
    if st.session_state.pop("_clear_keys", False):
        st.session_state.api_key_input = ""
        st.session_state.llm_key_input = ""
    st.subheader("1. 분석 설정")
    st.caption(key_status_text())
    st.text_input(
        "영상 URL 또는 영상 ID",
        value=DEFAULT_VIDEO_URL,
        key="video_input",
        help="기본값은 연구 대상 영상입니다.",
    )
    source = st.radio(
        "데이터 출처",
        ["YouTube API로 수집", "모의 데이터로 화면 확인"],
        key="data_source",
        help="모의 데이터는 실제 댓글이 아닙니다. API 키가 없을 때 화면과 분류 흐름을 확인하는 용도입니다.",
    )
    st.text_input(
        "YouTube API 키 (선택)",
        type="password",
        key="api_key_input",
        help="입력값은 이 세션 메모리에서만 사용합니다. 다운로드 파일과 로그에 저장하지 않습니다.",
    )
    columns = st.columns(2)
    with columns[0]:
        st.number_input(
            "최대 수집 수",
            min_value=0,
            max_value=20000,
            value=5000,
            step=100,
            key="max_comments",
            help="0을 입력하면 5,000건까지 가져옵니다. 상한은 20,000건입니다. 할당량을 우회하지는 않습니다.",
        )
    with columns[1]:
        st.checkbox("대댓글도 수집", value=True, key="include_replies")
    st.selectbox(
        "분류 방식",
        ["규칙 기반 (외부 전송 없음)", "외부 LLM (댓글 원문이 외부로 전송됨)", "로컬 ML 모델"],
        key="classifier_kind",
    )
    kind = st.session_state.get("classifier_kind") or ""
    if kind.startswith("외부"):
        st.warning("외부 LLM을 쓰면 댓글 원문이 선택한 서버로 전송됩니다. 기본값은 전송하지 않는 규칙 기반 분류입니다.")
        st.checkbox("댓글 원문을 외부 서비스로 보내는 데 동의합니다.", key="llm_consent")
        st.text_input("LLM API 키", type="password", key="llm_key_input")
        st.text_input("모델 이름", key="llm_model", placeholder="예: gpt-4o-mini")
        st.text_input("API 기본 주소 (비우면 OpenAI 호환 기본값)", key="llm_base")
    elif kind.startswith("로컬"):
        st.text_input("모델 파일 경로", key="ml_path", help="joblib으로 저장한 모델. 없으면 규칙 기반을 사용하세요.")
    else:
        st.caption("규칙 기반 분류는 이 컴퓨터 안에서만 계산하며 댓글을 외부로 보내지 않습니다.")
    if st.button("세션의 API 키 지우기", key="clear_key"):
        st.session_state._clear_keys = True
        st.rerun()
    clicked = st.button("분석 실행", type="primary", key="run_button")
    if clicked:
        try:
            if source.startswith("모의"):
                st.session_state.bundle = run_sample()
            else:
                st.session_state.bundle = run_api(
                    st.session_state.get("video_input") or "",
                    int(st.session_state.get("max_comments") or 0),
                    bool(st.session_state.get("include_replies")),
                )
        except UserFacingError as error:
            show_error(error)
    with st.expander("분류 기준 보기"):
        for item in ordered_categories():
            st.markdown(f"**{item['symbol']} {item['name']}**  \n{item['criteria']}")
        st.caption(CONFIDENCE_NOTE)


def render_collection(bundle: dict) -> None:
    st.subheader("2. 수집 상태")
    meta = bundle["meta"]
    report = bundle["report"]
    if meta.get("is_sample"):
        st.warning(meta.get("notice") or "모의 데이터입니다. 실제 유튜브 댓글이 아닙니다.")
    left, right = st.columns(2)
    with left:
        st.markdown(f"**영상 ID**  \n{meta.get('video_id')}")
        title = meta.get("video_title") or "제목을 가져오지 못했습니다."
        st.markdown(f"**영상 제목**  \n{title}")
    with right:
        st.markdown(f"**수집 시각**  \n{meta.get('collected_at')}")
        st.markdown(
            f"**페이지** {meta.get('pages') or 0} · **대댓글 추가 요청** {meta.get('reply_pages') or 0} · "
            f"**가져온 댓글** {report.get('raw_count', 0)}건"
        )
    if meta.get("api_comment_count") is not None:
        st.caption(f"API가 알려 준 댓글 수: {meta['api_comment_count']}건. 수집 수와 다를 수 있습니다.")
    if meta.get("warning"):
        st.info(meta["warning"])
    if meta.get("partial"):
        st.warning("수집이 끝까지 이어지지 않았습니다. 아래 결과는 받아 온 범위만 사용합니다.")


def render_summary(bundle: dict) -> None:
    st.subheader("3. 분석 요약")
    report = bundle["report"]
    stats = bundle["summary"]["stats"]
    columns = st.columns(4)
    columns[0].metric("수집 댓글", f"{report.get('raw_count', 0)}건")
    columns[1].metric("분류 대상 (분모)", f"{stats['total']}건")
    columns[2].metric("판단 보류", f"{stats['hold']}건")
    columns[3].metric("중복 + 제외", f"{report.get('duplicate_count', 0) + report.get('excluded_count', 0)}건")
    st.caption(
        f"중복 {report.get('duplicate_count', 0)}건을 제거했고, 제외 {report.get('excluded_count', 0)}건입니다. "
        f"비율의 분모는 분류 대상 {stats['total']}건이며 판단 보류를 포함합니다. "
        f"참고 점수 0.50 미만 또는 보류 {stats['low_confidence']}건, 사용자 수정 {stats['modified']}건."
    )
    reasons = report.get("exclusion_reasons") or {}
    if reasons:
        st.markdown("**제외 사유**")
        for reason, count in reasons.items():
            st.markdown(f"- {reason}: {count}건")
    with st.expander("제외·중복 기준과 해당 댓글"):
        for line in report.get("criteria") or []:
            st.markdown(f"- {line}")
        excluded = bundle.get("excluded") or []
        duplicates = bundle.get("duplicates") or []
        if not excluded and not duplicates:
            st.caption("제외되거나 중복으로 빠진 댓글이 없습니다.")
        for record in excluded:
            st.markdown(f"- ({record.get('exclude_reason')}) {record.get('text_analysis') or record.get('text_raw')}")
        for record in duplicates:
            st.markdown(f"- (중복) {record.get('text_analysis')}")


def show_html(document: str, height: int) -> None:
    if not document:
        return
    st.iframe(document, width="stretch", height=height)


def render_chart_and_table(bundle: dict) -> None:
    stats = bundle["summary"]["stats"]
    rows = stats["rows"]
    table = pd.DataFrame(
        [
            {
                "기호": row["기호"],
                "유형": row["유형"],
                "댓글 수 (건)": row["댓글수"],
                "비율 (%)": row["비율"],
            }
            for row in rows
        ]
    )
    st.markdown("**분류 유형별 댓글 수와 비율**")
    st.caption(
        f"막대 길이는 댓글 수입니다. 막대 끝의 비율은 분류 대상 {stats['total']}건을 분모로 계산했습니다. "
        "판단 보류도 분모에 포함됩니다. 반올림 때문에 비율 합이 100%가 아닐 수 있습니다. "
        "색만으로 구분하지 않도록 이름, 기호, 무늬, 범례를 함께 넣었습니다."
    )
    try:
        svg, png = render_category_chart(rows, stats["total"])
        del svg
        st.session_state.chart_png = png
        show_html(category_chart_html(rows, stats["total"]), 860)
        st.caption(font_caption() + " 화면의 막대는 글자 크기를 유지하는 HTML 차트이고, PNG 다운로드는 같은 글꼴로 그린 이미지입니다.")
    except UserFacingError as error:
        st.session_state.chart_png = b""
        show_error(error)
        show_html(category_chart_html(rows, stats["total"]), 860)
    st.markdown("**같은 수치를 표로 보기**")
    st.dataframe(table, width="stretch", hide_index=True)


def render_keywords(bundle: dict) -> None:
    keywords = bundle["summary"]["keywords"]
    status = analyzer_status()
    if status["ok"]:
        st.caption(status["message"] + " " + keywords["title_discount"])
    else:
        st.warning(status["message"])
    overall = pd.DataFrame(keywords["overall"]["rows"])
    st.markdown("**전체 키워드**")
    if keywords["overall"]["note"]:
        st.caption(keywords["overall"]["note"])
    if overall.empty:
        st.caption("반복된 키워드가 없습니다.")
    else:
        st.dataframe(overall, width="stretch", hide_index=True)
        show_html(keyword_chart_html(keywords["overall"]["rows"], "전체 댓글에서 자주 나타난 단어", "#0F5C8C"), 460)
    pairs = pd.DataFrame(keywords["collocations"])
    st.markdown("**자주 함께 나타난 단어**")
    if pairs.empty:
        st.caption("두 번 이상 이어진 단어 쌍이 없습니다. 자료가 적으면 이 표는 비어 있을 수 있습니다.")
    else:
        st.dataframe(pairs, width="stretch", hide_index=True)
    st.markdown("**유형별 키워드**")
    tabs = st.tabs([CATEGORIES[row["id"]]["name"] for row in bundle["summary"]["stats"]["rows"]])
    for tab, row in zip(tabs, bundle["summary"]["stats"]["rows"]):
        with tab:
            payload = keywords["by_category"][row["id"]]
            frame = pd.DataFrame(payload["rows"])
            if payload["note"]:
                st.caption(payload["note"])
            if frame.empty:
                st.caption("이 유형에서 보여줄 반복 단어가 없습니다.")
            else:
                st.dataframe(frame, width="stretch", hide_index=True)
                show_html(keyword_chart_html(payload["rows"], f"{row['라벨']}에서 자주 나타난 단어", row["색"]), 420)


def render_answers(bundle: dict) -> None:
    st.markdown("**연구 질문에 대한 답**")
    st.caption("수와 키워드는 이번 분류 결과에서 계산했습니다. 댓글 밖의 사실로 확대하지 않습니다.")
    for index, item in enumerate(bundle["summary"]["answers"], start=1):
        with st.container(border=True):
            st.markdown(f"**{index}. {item['question']}**")
            st.write(item["answer"])


def comment_card(record: dict) -> None:
    symbol = CATEGORIES[record["label"]]["symbol"]
    color = CATEGORIES[record["label"]]["color"]
    secondary = ", ".join(record.get("secondary_names") or []) or "없음"
    personal = ", ".join(record.get("personal_terms") or [])
    meta = (
        f"{symbol} {record['label_name']} · 참고 점수 {record['confidence']:.2f} · "
        f"좋아요 {record.get('like_count', 0)} · "
        f"{'대댓글' if record.get('is_reply') else '댓글'}"
    )
    if record.get("user_modified"):
        meta += " · 사용자 수정"
    if record.get("low_confidence"):
        meta += " · 검토 권장"
    evidence = "<br>".join(html.escape(line) for line in (record.get("evidence") or []))
    extra = f"<br>보조 태그: {html.escape(secondary)}"
    if personal:
        extra += f"<br>개인 상황 표현: {html.escape(personal)}"
    body = html.escape(record.get("text_analysis") or "").replace("\n", "<br>")
    st.markdown(
        f"""
        <div class="comment-card" style="border-left-color:{color};">
          <div class="meta">{html.escape(meta)}</div>
          <div class="body">{body}</div>
          <div class="meta" style="margin-top:0.45rem;">근거<br>{evidence}{extra}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_details(bundle: dict) -> None:
    st.subheader("5. 세부 결과")
    representatives = bundle["summary"]["representatives"]
    st.caption(representatives["criteria"])
    st.info(REPRESENTATIVE_LIMIT)
    by_category = representatives["by_category"]
    for item in ordered_categories():
        if item["id"] == "hold":
            continue
        picked = by_category.get(item["id"]) or []
        st.markdown(f"**{item['symbol']} {item['name']}** · {item['summary']}")
        if not picked:
            st.caption("이 유형으로 분류된 댓글이 없어 사례를 고르지 않았습니다.")
            continue
        for record in picked:
            comment_card(record)
    st.markdown("**판단 보류 예시**")
    st.caption("아래는 대표 사례가 아니라, 왜 유형을 정하지 않았는지 보여주는 글입니다.")
    if not representatives["hold_examples"]:
        st.caption("판단 보류 댓글이 없습니다.")
    for record in representatives["hold_examples"]:
        comment_card(record)
    st.markdown("**유형 간 비교**")
    st.caption("같은 순서와 기호를 사용합니다. 평균 참고 점수는 확률이 아닙니다.")
    st.dataframe(pd.DataFrame(bundle["summary"]["comparison"]), width="stretch", hide_index=True)
    replies = bundle["summary"]["replies"]
    st.markdown("**답글 관계**")
    st.write(replies["note"])
    if replies["rows"]:
        st.dataframe(pd.DataFrame(replies["rows"]), width="stretch", hide_index=True)
    timeline = bundle["summary"]["timeline"]
    st.markdown("**작성 시각 분포**")
    st.caption(timeline["note"])
    if timeline["rows"]:
        st.dataframe(pd.DataFrame(timeline["rows"]), width="stretch", hide_index=True)
    else:
        st.caption("작성 시각이 없어 분포를 만들지 않았습니다.")


def render_review(bundle: dict) -> None:
    st.subheader("6. 댓글 검토")
    st.caption("주 분류만 바꿀 수 있습니다. 저장 즉시 차트, 연구 질문 답, 다운로드 내용에 반영됩니다. 참고 점수는 수정 전 자동 분류 값입니다.")
    records = bundle["records"]
    filters = st.columns([2, 1, 1])
    with filters[0]:
        query = st.text_input("댓글 검색", key="review_query")
    with filters[1]:
        label_filter = st.selectbox("유형", ["전체", *CATEGORY_NAMES], key="review_label")
    with filters[2]:
        low_only = st.checkbox("검토 권장만", key="review_low")
    filtered = []
    for record in records:
        text = record.get("text_analysis") or ""
        if query and query.strip() not in text:
            continue
        if label_filter != "전체" and record.get("label_name") != label_filter:
            continue
        if low_only and not (record.get("low_confidence") or record.get("label") == "hold"):
            continue
        filtered.append(record)
    st.caption(f"표시 {len(filtered)}건 / 분류 대상 {len(records)}건")
    if not filtered:
        st.info("조건에 맞는 댓글이 없습니다.")
        return
    frame = pd.DataFrame(
        [
            {
                "번호": record["display_no"],
                "주 분류": record["label_name"],
                "원래 분류": record["original_label_name"],
                "보조 태그": ", ".join(record.get("secondary_names") or []) or "—",
                "참고 점수": record["confidence"],
                "검토": "수정됨" if record.get("user_modified") else ("검토 권장" if record.get("low_confidence") else ""),
                "대댓글": "예" if record.get("is_reply") else "아니오",
                "좋아요": record.get("like_count") or 0,
                "댓글": record.get("text_analysis") or "",
            }
            for record in filtered
        ]
    )
    edited = st.data_editor(
        frame,
        column_config={
            "주 분류": st.column_config.SelectboxColumn("주 분류", options=CATEGORY_NAMES, required=True),
            "댓글": st.column_config.TextColumn("댓글", width="large", disabled=True),
        },
        disabled=["번호", "원래 분류", "보조 태그", "참고 점수", "검토", "대댓글", "좋아요", "댓글"],
        hide_index=True,
        width="stretch",
        num_rows="fixed",
        height=460,
        key=f"editor_{label_filter}_{int(low_only)}_{query}",
    )
    changed = False
    by_no = {record["display_no"]: record for record in records}
    for _, row in edited.iterrows():
        record = by_no.get(int(row["번호"]))
        if record and row["주 분류"] != record["label_name"]:
            apply_user_label(record, name_to_id(str(row["주 분류"])))
            changed = True
    if changed:
        refresh_summary(bundle)
        st.rerun()
    choice = st.selectbox(
        "댓글 전문",
        options=[record["display_no"] for record in filtered],
        format_func=lambda number: f"{number}. {display_excerpt(by_no[number])}",
    )
    comment_card(by_no[choice])


def render_export(bundle: dict) -> None:
    st.subheader("7. 자료 내보내기")
    st.caption("CSV는 Excel에서 한글이 깨지지 않도록 UTF-8 BOM을 넣었습니다. JSON은 UTF-8이며 한글을 그대로 저장합니다. 작성자 이름, 채널, API 키는 포함하지 않습니다.")
    stamp = file_stamp(bundle["meta"].get("collected_at") or "", bundle["meta"].get("video_id") or "video")
    csv_bytes = analysis_csv_bytes(bundle["records"])
    json_bytes = analysis_json_bytes(bundle)
    excluded_bytes = excluded_csv_bytes(bundle.get("excluded") or [], bundle.get("duplicates") or [])
    columns = st.columns(3)
    columns[0].download_button(
        "분석 결과 CSV",
        data=csv_bytes,
        file_name=f"댓글분석_{stamp}.csv",
        mime="text/csv",
        width="stretch",
    )
    columns[1].download_button(
        "분석 결과 JSON",
        data=json_bytes,
        file_name=f"댓글분석_{stamp}.json",
        mime="application/json",
        width="stretch",
    )
    columns[2].download_button(
        "제외·중복 CSV",
        data=excluded_bytes,
        file_name=f"제외댓글_{stamp}.csv",
        mime="text/csv",
        width="stretch",
    )
    png = st.session_state.get("chart_png") or b""
    if png:
        st.download_button(
            "분류 차트 PNG",
            data=png,
            file_name=f"분류차트_{stamp}.png",
            mime="image/png",
            width="stretch",
        )


def render_limits(bundle: dict | None) -> None:
    st.subheader("8. 해석상 한계")
    limitations = (bundle or {}).get("summary", {}).get("limitations")
    if not limitations:
        from src.summarize import LIMITATIONS

        limitations = LIMITATIONS
    for line in limitations:
        st.markdown(f"- {line}")
    if bundle:
        st.markdown("**이번 분석의 결론**")
        st.write(bundle["summary"]["conclusion"])


def main() -> None:
    st.set_page_config(page_title="유튜브 댓글 분석기", page_icon="▣", layout="wide")
    inject_style()
    st.title("유튜브 댓글 분석기")
    st.write(
        "댓글에 드러난 반응, 기대, 사용 질문, 공감, 비판을 나누어 연구 질문에 답합니다. "
        "댓글을 남긴 사람은 전체 시청자를 대표하지 않으며, 주파수의 효과를 증명하는 자료가 아닙니다."
    )
    fonts = font_status()
    if not fonts["ready"]:
        st.error("한글 차트 글꼴을 찾지 못했습니다. 표로는 결과를 볼 수 있지만, 글꼴을 넣기 전에는 차트를 그리지 않습니다.")
    if os.environ.get("RUN_SAMPLE_ON_LOAD") == "1" and "bundle" not in st.session_state:
        st.session_state.classifier_kind = "규칙 기반 (외부 전송 없음)"
        st.session_state.bundle = run_sample()
    with st.container(border=True):
        render_settings()
    bundle = st.session_state.get("bundle")
    if not bundle:
        st.info("분석을 실행하면 수집 상태, 요약, 차트, 연구 질문 답이 이 아래에 나타납니다.")
        with st.container(border=True):
            render_limits(None)
        return
    with st.container(border=True):
        render_collection(bundle)
    with st.container(border=True):
        render_summary(bundle)
    with st.container(border=True):
        st.subheader("4. 핵심 결과")
        st.write(
            "아래 비율은 댓글을 남긴 작성자의 표현을 나눈 것입니다. "
            "영상을 본 사람 전체의 생각이나 주파수의 효과로 읽을 수 없습니다."
        )
        render_chart_and_table(bundle)
        render_keywords(bundle)
        render_answers(bundle)
    with st.container(border=True):
        render_details(bundle)
    with st.container(border=True):
        render_review(bundle)
    with st.container(border=True):
        render_export(bundle)
    with st.container(border=True):
        render_limits(bundle)
