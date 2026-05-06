from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from chatgpt_automation.batch import extract_json_fragment, load_jobs, parse_json_response
from chatgpt_automation.enrichment import apply_asset_llm_enrichment, normalize_llm_description


class BatchParsingTests(unittest.TestCase):
    def test_extract_json_fragment_from_code_block(self) -> None:
        text = "```json\n{\"short_caption\": \"BTC chart\"}\n```"
        self.assertEqual(extract_json_fragment(text), '{"short_caption": "BTC chart"}')

    def test_parse_json_response_from_mixed_text(self) -> None:
        text = "Here you go:\n{\"short_caption\": \"BTC chart\", \"confidence\": \"high\"}"
        payload = parse_json_response(text)
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["short_caption"], "BTC chart")

    def test_load_jobs_defaults_to_same_chat(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            jobs_path = Path(temp_dir) / "jobs.jsonl"
            jobs_path.write_text(
                '{"id":"job-1","prompt":"Describe the chart.","attachments":[]}\n'
                '{"id":"job-2","prompt":"Use a new chat.","attachments":[],"new_chat":true}\n',
                encoding="utf-8",
            )

            jobs = load_jobs(jobs_path)

        self.assertEqual(len(jobs), 2)
        self.assertFalse(jobs[0].new_chat)
        self.assertTrue(jobs[1].new_chat)


class EnrichmentTests(unittest.TestCase):
    def test_normalize_llm_description(self) -> None:
        payload = normalize_llm_description(
            {
                "short_caption": "  BTCUSDT H1 chart  ",
                "key_visual_elements": ["candles", "", "price scale"],
                "confidence": "HIGH",
            }
        )
        self.assertEqual(payload["short_caption"], "BTCUSDT H1 chart")
        self.assertEqual(payload["key_visual_elements"], ["candles", "price scale"])
        self.assertEqual(payload["confidence"], "high")

    def test_normalize_llm_description_repairs_mojibake_and_normalizes_quotes(self) -> None:
        mojibake_summary = "Label \u201cH1 BOS\u201d and \u2013 note".encode("utf-8").decode("latin-1")
        payload = normalize_llm_description(
            {
                "visual_summary": mojibake_summary,
                "limitations": ["\u201csmall text\u201d", "\u2013 cropped axis"],
            }
        )

        self.assertEqual(payload["visual_summary"], 'Label "H1 BOS" and - note')
        self.assertEqual(payload["limitations"], ['"small text"', "- cropped axis"])

    def test_apply_asset_llm_enrichment_updates_target_json(self) -> None:
        annotation = {
            "caption": "old caption",
            "summary": "old summary",
            "description": "old description",
            "clean_text": "old clean text",
            "extraction_methods": ["ocr"],
            "primary_symbol": "BTCUSDT",
            "timeframes": ["H1"],
            "venue": "BINANCE",
            "target_json": {
                "description": {
                    "short_caption": "old caption",
                    "visual_summary": "old description",
                    "context_augmented_summary": "old description",
                    "key_visual_elements": [],
                    "limitations": [],
                },
                "observed": {
                    "visible_in_crop": {
                        "clean_text": "old clean text",
                    }
                },
                "provenance": {
                    "extraction_methods": ["ocr"],
                },
            },
        }
        updated = apply_asset_llm_enrichment(
            annotation=annotation,
            response_payload={
                "short_caption": "BTCUSDT H1 candlestick chart",
                "visual_summary": "A trading chart with candles and a price scale.",
                "context_augmented_summary": "A BTCUSDT H1 chart with visible candles and price labels.",
                "key_visual_elements": ["candlesticks", "price_scale"],
                "limitations": ["small_text"],
                "visible_text": "BTCUSDT 1h",
                "training_tags": ["chart", "candles"],
                "confidence": "high",
            },
            raw_response_text='{"short_caption":"BTCUSDT H1 candlestick chart"}',
            prompt="describe image",
            language="en",
            model_slug="gpt-test",
            conversation_url="https://chatgpt.com/c/test",
        )
        self.assertEqual(updated["caption"], "BTCUSDT H1 candlestick chart")
        self.assertIn("chatgpt_browser_llm", updated["extraction_methods"])
        self.assertEqual(updated["target_json"]["description"]["key_visual_elements"], ["candlesticks", "price_scale"])
        self.assertIn("Short caption:", updated["clean_text"])
        self.assertEqual(updated["target_json"]["observed"]["visible_in_crop"]["clean_text"], "old clean text")
        self.assertEqual(updated["llm_enrichment"]["model_slug"], "gpt-test")

    def test_apply_asset_llm_enrichment_preserves_existing_visual_summary(self) -> None:
        annotation = {
            "caption": "old caption",
            "summary": "old summary",
            "description": "old context summary",
            "clean_text": "old clean text",
            "extraction_methods": ["ocr"],
            "target_json": {
                "description": {
                    "short_caption": "old caption",
                    "visual_summary": "existing visual summary",
                    "context_augmented_summary": "existing context summary",
                    "key_visual_elements": ["chart_panel"],
                    "limitations": ["small_text"],
                },
                "observed": {
                    "visible_in_crop": {
                        "clean_text": "crop only clean text",
                    }
                },
                "provenance": {
                    "extraction_methods": ["ocr"],
                },
            },
        }

        updated = apply_asset_llm_enrichment(
            annotation=annotation,
            response_payload={
                "short_caption": "new caption",
                "context_augmented_summary": "new context summary",
                "confidence": "medium",
            },
            raw_response_text='{"short_caption":"new caption"}',
            prompt="describe image",
            language="en",
            model_slug="gpt-test",
            conversation_url="https://chatgpt.com/c/test",
        )

        self.assertEqual(updated["target_json"]["description"]["visual_summary"], "existing visual summary")
        self.assertEqual(updated["target_json"]["description"]["key_visual_elements"], ["chart_panel"])
        self.assertEqual(updated["target_json"]["description"]["limitations"], ["small_text"])
        self.assertEqual(updated["target_json"]["observed"]["visible_in_crop"]["clean_text"], "crop only clean text")

    def test_apply_asset_llm_enrichment_upgrades_legacy_v2_description_shape(self) -> None:
        annotation = {
            "annotation_version": "2.0",
            "caption": "legacy caption",
            "summary": "legacy summary",
            "description": "annotation fallback description",
            "clean_text": "legacy clean text",
            "extraction_methods": ["ocr"],
            "target_json": {
                "description": "legacy target description",
                "observed": {
                    "visible_in_crop": {
                        "clean_text": "legacy crop clean text",
                    }
                },
                "provenance": {
                    "extraction_methods": ["ocr"],
                },
            },
        }

        updated = apply_asset_llm_enrichment(
            annotation=annotation,
            response_payload={
                "short_caption": "new caption",
                "confidence": "medium",
            },
            raw_response_text='{"short_caption":"new caption"}',
            prompt="describe image",
            language="en",
            model_slug="gpt-test",
            conversation_url="https://chatgpt.com/c/test",
        )

        self.assertEqual(updated["target_json"]["description"]["short_caption"], "new caption")
        self.assertEqual(updated["target_json"]["description"]["context_augmented_summary"], "legacy target description")
        self.assertEqual(updated["target_json"]["description"]["visual_summary"], "")
        self.assertEqual(updated["target_json"]["observed"]["visible_in_crop"]["clean_text"], "legacy crop clean text")


if __name__ == "__main__":
    unittest.main()
