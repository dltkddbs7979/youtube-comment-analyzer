# -*- coding: utf-8 -*-
import json
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.classifier import apply_user_label, classify_text
from src.export_data import analysis_csv_bytes, analysis_json_bytes
from src.fonts_setup import bundled_font_path
from src.pipeline import refresh_summary, run_prepared
from src.sample_data import load_sample_comments, load_sample_file
from src.summarize import FORBIDDEN_CONCLUSION
from src.youtube_api import map_api_error, parse_video_id, validate_api_key_format
from src.errors import UserFacingError


def _sample_bundle():
    comments, notice, title = load_sample_comments()
    return run_prepared(
        comments,
        {
            "video_id": "15Gbo7Xcy80",
            "video_title": title,
            "is_sample": True,
            "notice": notice,
            "source": "sample",
            "collected_at": "2026-04-01T12:00:00+09:00",
        },
    )


class ClassificationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.bundle = _sample_bundle()
        cls.by_id = {record["comment_id"]: record for record in cls.bundle["records"]}

    def test_sample_loader_hides_expected_labels(self):
        comments, notice, _title = load_sample_comments()
        self.assertIn("모의 데이터", notice)
        self.assertTrue(all("_expected" not in item for item in comments))

    def test_expected_labels(self):
        raw = load_sample_file()
        mismatches = []
        for item in raw["comments"]:
            expected = item["_expected"]
            if expected in {"duplicate", "excluded"}:
                self.assertNotIn(item["comment_id"], self.by_id)
                continue
            got = self.by_id[item["comment_id"]]["label"]
            if got != expected:
                mismatches.append((item["comment_id"], expected, got, self.by_id[item["comment_id"]]["evidence"]))
        self.assertEqual(mismatches, [])

    def test_wish_is_not_experience(self):
        result = classify_text("연락 왔으면 좋겠다")
        self.assertEqual(result.label, "hope")
        self.assertNotEqual(result.label, "effect")

    def test_vague_effect_claim_is_not_experience(self):
        result = classify_text("효과 있어요")
        self.assertNotEqual(result.label, "effect")

    def test_negated_contact_is_not_experience(self):
        result = classify_text("아직 연락은 안 왔어요.")
        self.assertNotEqual(result.label, "effect")

    def test_short_laughter_is_humor_with_low_confidence(self):
        result = classify_text("ㅋㅋㅋㅋ")
        self.assertEqual(result.label, "humor")
        self.assertLess(result.confidence, 0.5)

    def test_waiting_is_hope_not_experience(self):
        result = classify_text("아직 연락은 안 왔어요.")
        self.assertEqual(result.label, "hope")
        self.assertNotEqual(result.label, "effect")

    def test_ordinary_comments_are_not_held(self):
        samples = [
            "공부하면서 듣고 있어요",
            "오늘도 화이팅",
            "ㅋㅋㅋㅋ",
            "좋아요",
            "❤️",
            "이어폰으로 들으면 되나요",
            "연락 왔으면 좋겠다",
            "사기 같아요",
            "아직 답장이 없어요",
            "재회하고 싶어요",
            "대박",
            "그냥 한번 들어봤어요",
            "효과 있어요",
            "3시간째 듣는 중",
            "ㅠㅠ",
            "집중이 잘 되네요",
            "별로예요",
            "그 사람 생각하면서 들어요",
        ]
        held = [text for text in samples if classify_text(text).label == "hold"]
        self.assertEqual(held, [])

    def test_confidence_is_not_presented_as_certainty(self):
        result = classify_text("사흘 동안 자기 전에 들었는데, 읽씹만 하던 사람에게서 먼저 연락이 왔어요.")
        self.assertLessEqual(result.confidence, 0.92)
        self.assertTrue(any("확률" in line for line in result.evidence))

    def test_preprocess_masks_url_entity_phone_and_counts(self):
        report = self.bundle["report"]
        self.assertEqual(report["duplicate_count"], 1)
        self.assertGreaterEqual(report["excluded_count"], 2)
        self.assertNotIn("c33", self.by_id)
        entity = self.by_id["c36"]
        self.assertNotIn("&amp;", entity["text_raw"])
        self.assertIn("&", entity["text_raw"])
        linked = self.by_id["c37"]
        self.assertIn("[URL]", linked["text_analysis"])
        self.assertNotIn("example.com", linked["text_analysis"])
        self.assertIn("example.com", linked["text_raw"])
        phone = self.by_id["c41"]
        self.assertIn("[전화번호]", phone["text_analysis"])
        self.assertNotIn("010-1234-5678", phone["text_raw"])
        self.assertNotIn("010-1234-5678", json.dumps(self.bundle["summary"]["conclusion"]))

    def test_representative_is_not_chosen_by_likes_alone(self):
        picked = self.bundle["summary"]["representatives"]["by_category"]["effect"]
        self.assertTrue(picked)
        self.assertNotEqual(picked[0]["comment_id"], "c40")
        self.assertLessEqual(self.by_id["c40"]["confidence"], 0.60)

    def test_conclusion_avoids_causal_claims(self):
        text = self.bundle["summary"]["conclusion"]
        self.assertIn("모의 데이터", text)
        self.assertIn("판단 보류", text)
        for forbidden in FORBIDDEN_CONCLUSION:
            self.assertNotIn(forbidden, text)
        for answer in self.bundle["summary"]["answers"]:
            for forbidden in FORBIDDEN_CONCLUSION:
                self.assertNotIn(forbidden, answer["answer"])

    def test_user_edit_is_exported(self):
        bundle = _sample_bundle()
        target = next(record for record in bundle["records"] if record["comment_id"] == "c05")
        apply_user_label(target, "usage")
        refresh_summary(bundle)
        csv_text = analysis_csv_bytes(bundle["records"]).decode("utf-8-sig")
        self.assertIn("사용 방법 질문형", csv_text)
        document = json.loads(analysis_json_bytes(bundle).decode("utf-8"))
        labels = [item["label"] for item in document["comments"] if "왔으면" in item["text_analysis"]]
        self.assertIn("사용 방법 질문형", labels)
        self.assertNotIn("author", json.dumps(document))
        self.assertNotIn("api_key", json.dumps(document).lower())


class ExportAndApiTests(unittest.TestCase):
    def test_csv_has_bom_and_korean(self):
        bundle = _sample_bundle()
        raw = analysis_csv_bytes(bundle["records"])
        self.assertTrue(raw.startswith(b"\xef\xbb\xbf"))
        text = raw.decode("utf-8-sig")
        self.assertIn("댓글원문", text)
        self.assertIn("효과 경험형", text)
        self.assertNotIn("\\u", text)

    def test_json_keeps_korean_characters(self):
        bundle = _sample_bundle()
        raw = analysis_json_bytes(bundle)
        text = raw.decode("utf-8")
        self.assertIn("기대·희망형", text)
        self.assertNotIn("\\uc720", text)
        self.assertNotRegex(text, r"\\u[0-9a-fA-F]{4}")

    def test_video_id_parser(self):
        self.assertEqual(parse_video_id("15Gbo7Xcy80"), "15Gbo7Xcy80")
        self.assertEqual(parse_video_id("https://www.youtube.com/watch?v=15Gbo7Xcy80"), "15Gbo7Xcy80")
        self.assertEqual(parse_video_id("https://youtu.be/15Gbo7Xcy80"), "15Gbo7Xcy80")
        with self.assertRaises(UserFacingError):
            parse_video_id("https://example.com/not-a-video")

    def test_api_errors_do_not_leak_key(self):
        secret = "AIzaSECRETKEY1234567890"
        error = map_api_error(
            400,
            {"error": {"message": f"API key {secret} not valid", "errors": [{"reason": "keyInvalid"}]}},
        )
        self.assertNotIn(secret, error.as_text())
        self.assertNotIn("not valid", error.as_text())
        with self.assertRaises(UserFacingError) as caught:
            validate_api_key_format("bad " + secret)
        self.assertNotIn(secret, caught.exception.as_text())

    def test_bundled_font_has_korean_glyphs(self):
        path = bundled_font_path()
        self.assertIsNotNone(path)
        from fontTools.ttLib import TTFont

        font = TTFont(path)
        cmap = font.getBestCmap()
        for char in "댓글분석효과경험판단보류주파수연애":
            self.assertIn(ord(char), cmap, char)

    def test_chart_png_is_not_empty(self):
        from src.charts import render_category_chart

        bundle = _sample_bundle()
        _svg, png = render_category_chart(bundle["summary"]["stats"]["rows"], bundle["summary"]["stats"]["total"])
        self.assertGreater(len(png), 10_000)
        self.assertTrue(png.startswith(b"\x89PNG"))


if __name__ == "__main__":
    unittest.main()
