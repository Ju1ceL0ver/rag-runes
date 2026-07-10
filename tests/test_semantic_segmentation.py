from __future__ import annotations

import unittest
import sys
import types

if "requests" not in sys.modules:
    requests_stub = types.ModuleType("requests")

    def _missing_post(*args, **kwargs):  # noqa: ANN002, ANN003
        raise RuntimeError("requests is not installed in this test environment")

    requests_stub.post = _missing_post
    sys.modules["requests"] = requests_stub

from rag_runes.ingest.semantic_segmenter import (
    LLMSemanticSegmenter,
    RuleBasedSemanticSegmenter,
)
from rag_runes.ingest.tree_builder import detect_heading, is_page_noise
from rag_runes.ocr.glm_ocr import GLMOCRClient
from rag_runes.text_utils import normalize_whitespace


class RejectingLLMSegmenter(LLMSemanticSegmenter):
    def _request(self, payload):  # noqa: ANN001
        return {
            "choices": [
                {
                    "message": {
                        "content": '{"chunks":["A short summary that is not copied from the text."]}'
                    }
                }
            ]
        }


class SemanticSegmentationTests(unittest.TestCase):
    def test_rule_based_split_keeps_topic_blocks_and_bounds(self) -> None:
        text = "\n".join(
            [
                "The Khazarian Rovas script was used across several relic groups.",
                "Figure 2-1: Drawing of an inscription from Khumara fortress.",
                "Written in Khazarian Rovas Font",
                "IPA phonetic transcription tuɣalas d͡ʒapdi biʧigi.",
                "Translation from Ogur Tughalas built and wrote its inscription.",
            ]
        )

        chunks = RuleBasedSemanticSegmenter().split(text, max_chars=170, overlap=0)

        self.assertGreaterEqual(len(chunks), 2)
        self.assertTrue(all(len(chunk) <= 170 for chunk in chunks))
        self.assertIn("Figure 2-1", " ".join(chunks))
        self.assertIn("Translation from Ogur", " ".join(chunks))

    def test_llm_split_rejects_non_source_summary_and_keeps_tail(self) -> None:
        text = "\n".join(
            [
                "The first paragraph explains Khazarian Rovas inscriptions and their evidence. "
                * 4,
                "Figure 2-1: Drawing of an inscription from Khumara fortress.",
                "TAIL_UNIQUE sentence about character ordering and punctuation.",
            ]
        )
        segmenter = RejectingLLMSegmenter(
            endpoint="http://127.0.0.1:9/v1/chat/completions",
            model="fake",
            max_input_chars=220,
        )

        chunks = segmenter.split(text, max_chars=150, overlap=0)
        joined = normalize_whitespace(" ".join(chunks))

        self.assertNotIn("short summary", joined)
        self.assertIn("TAIL_UNIQUE", joined)
        self.assertTrue(all(len(chunk) <= 150 for chunk in chunks))

    def test_source_validation_requires_direct_coverage(self) -> None:
        segmenter = LLMSemanticSegmenter(endpoint="http://example.invalid", model="fake")
        source = "Alpha sentence. Beta sentence. Gamma sentence."

        self.assertTrue(
            segmenter._source_split_is_valid(
                source,
                ["Alpha sentence. Beta sentence.", "Gamma sentence."],
                overlap=0,
            )
        )
        self.assertFalse(
            segmenter._source_split_is_valid(
                source,
                ["Summary of alpha beta gamma."],
                overlap=0,
            )
        )

    def test_heading_detection_rejects_wrapped_body_lines_and_toc_entries(self) -> None:
        wrapped_body = (
            "CE) in the Carpathian Basin in the 10th century by the Hungarians, "
            "Kavars (Khazar subjects rebelled against the"
        )

        self.assertIsNone(detect_heading(wrapped_body))
        self.assertIsNone(
            detect_heading(
                "3. Unicode Character Properties ................................................ 8"
            )
        )
        self.assertIsNone(detect_heading("4 Vásáry, 2003; Erdélyi, 2004, p. 76"))
        self.assertIsNone(detect_heading("L2/11-089"))
        self.assertIsNone(detect_heading("X nªb ÿx"))
        self.assertIsNone(detect_heading("I. M"))
        self.assertIsNone(
            detect_heading("1. Has this proposal for addition been submitted before?")
        )
        self.assertIsNone(detect_heading("1. /"))
        self.assertIsNone(detect_heading("2. У"))
        self.assertIsNone(
            detect_heading("1xx00; b KHAZARIAN ROVAS LETTER ANGLED B")
        )
        self.assertEqual(detect_heading("I. Introduction"), (1, "Introduction"))
        self.assertEqual(
            detect_heading("C. Technical – Justification"),
            (2, "C. Technical – Justification"),
        )
        self.assertEqual(
            detect_heading(
                "6. The context of use for the proposed characters (type of use; common or rare)"
            ),
            (
                2,
                "The context of use for the proposed characters (type of use; common or rare)",
            ),
        )
        self.assertEqual(detect_heading("KHAZARIAN ROVAS"), (2, "KHAZARIAN ROVAS"))
        self.assertEqual(
            detect_heading("3. Unicode Character Properties"),
            (1, "Unicode Character Properties"),
        )

    def test_page_noise_and_empty_ocr_cleanup(self) -> None:
        self.assertTrue(
            is_page_noise("Proposal for encoding the Khazarian Rovas script in the SMP - 2 / 19")
        )
        self.assertEqual(GLMOCRClient._clean_response_text("```markdown\n```"), "")
        self.assertEqual(
            GLMOCRClient._clean_response_text("```markdown\nRune text\n```"),
            "Rune text",
        )


if __name__ == "__main__":
    unittest.main()
