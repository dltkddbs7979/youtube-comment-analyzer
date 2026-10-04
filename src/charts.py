# -*- coding: utf-8 -*-
"""한글 글꼴을 글리프로 넣어 그리는 차트.

브라우저 기본 글꼴에만 기대지 않는다.
글꼴이 없으면 빈 차트를 만들지 않고 오류를 돌려준다.
"""

from __future__ import annotations

from io import BytesIO, StringIO
import html as html_lib
import re

from src.errors import UserFacingError
from src.fonts_setup import bundled_font_path, chart_font_path, missing_font_message

_FONT_READY = False


def _ensure_font():
    global _FONT_READY
    path = chart_font_path()
    if path is None:
        title, detail, hint = missing_font_message()
        raise UserFacingError(title, detail, hint, code="font")
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import font_manager
    import matplotlib.pyplot as plt

    if not _FONT_READY:
        font_manager.fontManager.addfont(str(path))
        name = font_manager.FontProperties(fname=str(path)).get_name()
        plt.rcParams["font.family"] = name
        plt.rcParams["axes.unicode_minus"] = False
        plt.rcParams["svg.fonttype"] = "path"
        _FONT_READY = True
    return path


def font_caption() -> str:
    path = chart_font_path()
    if path is None:
        return "차트 글꼴을 찾지 못했습니다."
    if bundled_font_path() is not None and path == bundled_font_path():
        return "차트 글자: IBM Plex Sans KR. 프로젝트에 포함한 글꼴로 글리프를 그려 한글이 네모로 바뀌지 않게 했습니다."
    return f"번들 글꼴이 없어 시스템 글꼴({path.name})로 그렸습니다. 가능하면 IBM Plex Sans KR 파일을 다시 두세요."


def _finish(fig, png: bool = True) -> tuple[str, bytes]:
    import matplotlib.pyplot as plt

    svg_buffer = StringIO()
    fig.savefig(svg_buffer, format="svg", bbox_inches="tight", facecolor="white")
    svg = svg_buffer.getvalue()
    png_bytes = b""
    if png:
        png_buffer = BytesIO()
        fig.savefig(png_buffer, format="png", dpi=140, bbox_inches="tight", facecolor="white")
        png_bytes = png_buffer.getvalue()
    plt.close(fig)
    return svg, png_bytes


def render_category_chart(rows: list[dict], total: int) -> tuple[str, bytes]:
    _ensure_font()
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    if not rows:
        raise UserFacingError(
            "그릴 분류 결과가 없습니다",
            "분류 대상 댓글이 없어 막대그래프를 만들지 않았습니다.",
            "수집 결과와 제외 사유를 확인하세요.",
            code="empty_chart",
        )
    labels = [row["라벨"] for row in rows]
    counts = [row["댓글수"] for row in rows]
    colors = [row["색"] for row in rows]
    fig, ax = plt.subplots(figsize=(11.4, 6.8))
    fig.patch.set_facecolor("white")
    positions = list(range(len(rows)))
    hatches = []
    from src.categories import CATEGORIES

    for row in rows:
        hatches.append(CATEGORIES[row["id"]]["hatch"])
    bars = ax.barh(
        positions,
        counts,
        color=colors,
        edgecolor="#243140",
        linewidth=0.6,
        height=0.68,
        zorder=2,
    )
    for bar, hatch in zip(bars, hatches):
        bar.set_hatch(hatch)
    maximum = max(counts) if counts else 1
    ax.set_xlim(0, max(maximum * 1.38, 1))
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=14)
    ax.invert_yaxis()
    for index, row in enumerate(rows):
        ax.text(
            row["댓글수"] + max(maximum * 0.02, 0.08),
            index,
            f"{row['댓글수']}건 ({row['비율']:.1f}%)",
            va="center",
            ha="left",
            fontsize=13,
            color="#142033",
        )
    ax.set_xlabel("댓글 수 (건)", fontsize=14)
    ax.set_ylabel("분류 유형", fontsize=14)
    ax.set_title("분류 유형별 댓글 수와 비율", loc="left", fontsize=18, color="#142033", pad=16)
    subtitle = f"비율의 분모: 분류 대상 {total}건 (중복·제외 후, 판단 보류 포함)"
    ax.text(0, 1.02, subtitle, transform=ax.transAxes, fontsize=12, color="#3E4A59", va="bottom")
    ax.xaxis.grid(True, color="#E1E8F0", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelsize=12)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    handles = [
        Patch(facecolor=row["색"], edgecolor="#243140", label=row["라벨"], hatch=CATEGORIES[row["id"]]["hatch"])
        for row in rows
    ]
    ax.legend(
        handles=handles,
        title="범례 (색과 무늬, 이름)",
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncol=2,
        frameon=False,
        fontsize=12,
        title_fontsize=12,
    )
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.28, top=0.86, left=0.24)
    return _finish(fig)


def render_keyword_chart(rows: list[dict], title: str, color: str) -> tuple[str, bytes] | None:
    if not rows:
        return None
    _ensure_font()
    import matplotlib.pyplot as plt

    show = list(reversed(rows[:8]))
    fig, ax = plt.subplots(figsize=(10.2, 4.8))
    fig.patch.set_facecolor("white")
    positions = list(range(len(show)))
    counts = [row["빈도"] for row in show]
    ax.barh(positions, counts, color=color, edgecolor="#243140", height=0.66, zorder=2)
    maximum = max(counts) if counts else 1
    ax.set_xlim(0, maximum * 1.25)
    ax.set_yticks(positions)
    ax.set_yticklabels([row["단어"] for row in show], fontsize=13)
    for index, row in enumerate(show):
        ax.text(row["빈도"] + maximum * 0.02, index, f"{row['빈도']}회", va="center", fontsize=12, color="#142033")
    ax.set_xlabel("등장 댓글 수 (회)", fontsize=13)
    ax.set_ylabel("단어", fontsize=13)
    ax.set_title(title, loc="left", fontsize=16, color="#142033", pad=12)
    ax.xaxis.grid(True, color="#E1E8F0", zorder=0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    return _finish(fig, png=False)


def category_chart_html(rows: list[dict], total: int) -> str:
    """화면용 막대그래프. 글자 크기를 유지하고 좁은 화면에서는 줄을 나눈다."""
    from src.categories import CATEGORIES

    maximum = max((row["댓글수"] for row in rows), default=0)
    maximum = max(maximum, 1)
    bars = []
    legends = []
    for row in rows:
        width = 0 if row["댓글수"] == 0 else max(4, round(row["댓글수"] / maximum * 100))
        hatch = _hatch_css(CATEGORIES[row["id"]]["hatch"], row["색"])
        label = html_lib.escape(row["라벨"])
        count = row["댓글수"]
        ratio = row["비율"]
        bars.append(
            f"""
            <div class="row">
              <div class="name">{label}</div>
              <div class="track" role="img" aria-label="{label} {count}건 {ratio:.1f}퍼센트">
                <div class="bar" style="width:{width}%;{hatch}"></div>
              </div>
              <div class="num">{count}건 ({ratio:.1f}%)</div>
            </div>
            """
        )
        legends.append(
            f'<span class="legend-item"><i style="{hatch}"></i>{row["라벨"]}</span>'
        )
    body = "\n".join(bars)
    legend = "\n".join(legends)
    return _chart_document(
        f"""
        <h2>분류 유형별 댓글 수와 비율</h2>
        <p class="sub">비율의 분모: 분류 대상 {total}건 (중복·제외 후, 판단 보류 포함). 막대 길이는 댓글 수(건)에 비례합니다.</p>
        {body}
        <p class="axis">댓글 수 (건) · 분류 유형은 왼쪽 이름입니다.</p>
        <div class="legend"><div class="legend-title">범례 (색, 무늬, 이름)</div>{legend}</div>
        """
    )


def keyword_chart_html(rows: list[dict], title: str, color: str) -> str:
    if not rows:
        return ""
    safe_title = html_lib.escape(title)
    show = rows[:8]
    maximum = max(row["빈도"] for row in show) or 1
    bars = []
    for row in show:
        width = max(4, round(row["빈도"] / maximum * 100))
        word = html_lib.escape(str(row["단어"]))
        bars.append(
            f"""
            <div class="row">
              <div class="name">{word}</div>
              <div class="track"><div class="bar" style="width:{width}%;background:{color};"></div></div>
              <div class="num">{row["빈도"]}회</div>
            </div>
            """
        )
    return _chart_document(
        f"""
        <h2>{safe_title}</h2>
        <p class="sub">막대 길이는 그 단어가 나타난 댓글 수입니다. 단위는 회입니다.</p>
        {''.join(bars)}
        <p class="axis">등장 댓글 수 (회)</p>
        """
    )


def _hatch_css(hatch: str, color: str) -> str:
    overlays = {
        "//": "repeating-linear-gradient(45deg, transparent, transparent 4px, rgba(20,32,51,.28) 4px, rgba(20,32,51,.28) 6px)",
        "..": "radial-gradient(rgba(20,32,51,.45) 1.1px, transparent 1.3px)",
        "xx": "repeating-linear-gradient(45deg, transparent, transparent 4px, rgba(20,32,51,.28) 4px, rgba(20,32,51,.28) 6px), repeating-linear-gradient(-45deg, transparent, transparent 4px, rgba(20,32,51,.28) 4px, rgba(20,32,51,.28) 6px)",
        "++": "repeating-linear-gradient(0deg, transparent, transparent 4px, rgba(20,32,51,.28) 4px, rgba(20,32,51,.28) 6px), repeating-linear-gradient(90deg, transparent, transparent 4px, rgba(20,32,51,.28) 4px, rgba(20,32,51,.28) 6px)",
        "oo": "radial-gradient(circle at 4px 4px, transparent 2.2px, rgba(20,32,51,.35) 2.5px, transparent 3px)",
        "--": "repeating-linear-gradient(0deg, transparent, transparent 5px, rgba(255,255,255,.55) 5px, rgba(255,255,255,.55) 7px)",
    }
    overlay = overlays.get(hatch)
    if not overlay:
        return f"background:{color};"
    size = "background-size: 8px 8px;" if hatch in {"..", "oo"} else ""
    return f"background-color:{color};background-image:{overlay};{size}"


def _chart_document(inner: str) -> str:
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <style>
    html, body {{
      margin: 0;
      background: #ffffff;
      color: #142033;
      font-family: "Malgun Gothic", "맑은 고딕", "Apple SD Gothic Neo", "Noto Sans KR", "IBM Plex Sans KR", sans-serif;
    }}
    body {{ padding: 8px 4px 12px; }}
    h2 {{ font-size: 18px; line-height: 1.4; margin: 0 0 6px; }}
    .sub, .axis {{ font-size: 14px; line-height: 1.5; color: #3E4A59; margin: 0 0 12px; }}
    .row {{
      display: grid;
      grid-template-columns: minmax(8.5rem, 12rem) minmax(0, 1fr) auto;
      gap: 8px 10px;
      align-items: center;
      margin: 0 0 10px;
    }}
    .name {{ font-size: 15px; line-height: 1.35; overflow-wrap: anywhere; }}
    .track {{
      height: 22px;
      background: #F4F7FB;
      border-radius: 4px;
      overflow: hidden;
    }}
    .bar {{ height: 100%; border-radius: 4px; min-width: 0; }}
    .num {{ font-size: 15px; white-space: nowrap; }}
    .legend {{ display: flex; flex-wrap: wrap; gap: 8px 14px; margin-top: 8px; }}
    .legend-title {{ width: 100%; font-size: 14px; color: #3E4A59; }}
    .legend-item {{ display: inline-flex; align-items: center; gap: 6px; font-size: 14px; }}
    .legend-item i {{
      width: 18px; height: 12px; display: inline-block; border-radius: 2px; border: 1px solid #243140;
    }}
    @media (max-width: 680px) {{
      .row {{ grid-template-columns: minmax(0, 1fr) auto; }}
      .name {{ grid-column: 1 / -1; }}
    }}
  </style>
</head>
<body>
{inner}
</body>
</html>"""
    body = svg
    if "<?xml" in body[:80]:
        body = re.sub(r"^<\?xml[^>]*>\s*", "", body, count=1)
    body = re.sub(r'\swidth="[^"]+"', ' width="100%"', body, count=1)
    body = re.sub(r'\sheight="[^"]+"', ' height="auto"', body, count=1)
    return f"""<!DOCTYPE html>
<html lang="ko">
<head>
  <meta charset="utf-8">
  <style>
    html, body {{ margin: 0; background: #ffffff; }}
    .scroll {{ width: 100%; overflow-x: auto; }}
    .sheet {{ min-width: {min_width}px; }}
    svg {{ display: block; }}
  </style>
</head>
<body>
  <div class="scroll"><div class="sheet">{body}</div></div>
</body>
</html>"""
