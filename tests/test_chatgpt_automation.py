from __future__ import annotations

import json
import shutil
import sys
import unittest
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from chatgpt_automation.batch import BatchJob, extract_json_fragment, load_jobs, parse_json_response, run_batch_jobs
from chatgpt_automation.client import ChatGPTResponse
from chatgpt_automation.enrichment import (
    apply_asset_llm_enrichment,
    build_multimodal_description_prompt,
    normalize_llm_description,
)
from enrich_multimodal_descriptions import collect_annotation_jobs


class BatchParsingTests(unittest.TestCase):
    def test_extract_json_fragment_from_code_block(self) -> None:
        text = "```json\n{\"short_caption\": \"BTC chart\"}\n```"
        self.assertEqual(extract_json_fragment(text), '{"short_caption": "BTC chart"}')

    def test_parse_json_response_from_mixed_text(self) -> None:
        text = "Here you go:\n{\"short_caption\": \"BTC chart\", \"confidence\": \"high\"}"
        payload = parse_json_response(text)
        self.assertIsInstance(payload, dict)
        self.assertEqual(payload["short_caption"], "BTC chart")

    def test_extract_json_fragment_ignores_braces_inside_strings(self) -> None:
        text = 'Result:\n{"visual_summary":"Highlighted zone {A} near top","confidence":"high"}'
        self.assertEqual(
            extract_json_fragment(text),
            '{"visual_summary":"Highlighted zone {A} near top","confidence":"high"}',
        )

    def test_parse_json_response_repairs_unescaped_inner_quotes(self) -> None:
        text = (
            '{"visual_summary":"A chart with arrows labeled "H1 BOS" indicating structure breaks.",'
            '"confidence":"high"}'
        )

        payload = parse_json_response(text)

        self.assertIsInstance(payload, dict)
        self.assertEqual(
            payload["visual_summary"],
            'A chart with arrows labeled "H1 BOS" indicating structure breaks.',
        )
        self.assertEqual(payload["confidence"], "high")

    def test_load_jobs_defaults_to_same_chat(self) -> None:
        temp_root = ROOT / ".tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        jobs_path = temp_root / f"test_load_jobs_{uuid.uuid4().hex}.jsonl"
        try:
            jobs_path.write_text(
                '{"id":"job-1","prompt":"Describe the chart.","attachments":[]}\n'
                '{"id":"job-2","prompt":"Use a new chat.","attachments":[],"new_chat":true}\n',
                encoding="utf-8",
            )

            jobs = load_jobs(jobs_path)
        finally:
            if jobs_path.exists():
                jobs_path.unlink()

        self.assertEqual(len(jobs), 2)
        self.assertFalse(jobs[0].new_chat)
        self.assertTrue(jobs[1].new_chat)

    def test_run_batch_jobs_marks_invalid_json_as_error(self) -> None:
        class StubClient:
            def run_prompt(
                self,
                prompt: str,
                attachments: list[Path],
                new_chat: bool,
                allow_manual_login: bool,
            ) -> ChatGPTResponse:
                return ChatGPTResponse(
                    message_id="msg-1",
                    model_slug="gpt-test",
                    text="This is not valid JSON.",
                    url="https://chatgpt.com/c/test",
                )

        rows = run_batch_jobs(
            client=StubClient(),
            jobs=[
                BatchJob(
                    job_id="job-1",
                    prompt="Describe the chart.",
                    attachments=(),
                    metadata={},
                )
            ],
        )

        self.assertEqual(rows[0]["status"], "error")
        self.assertEqual(rows[0]["error"], "ChatGPT response did not contain valid JSON.")
        self.assertIsNone(rows[0]["assistant_json"])


class EnrichmentJobTests(unittest.TestCase):
    def test_collect_annotation_jobs_forces_fresh_chat_when_max_assets_per_chat_is_one(self) -> None:
        temp_root = ROOT / ".tmp" / f"test_collect_annotation_jobs_{uuid.uuid4().hex}"
        annotation_dir = temp_root / "annotations" / "03-10" / "assets"
        image_dir = temp_root / "images" / "03-10" / "assets"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        annotation_path = annotation_dir / "page_0001_asset_01.json"
        image_path = image_dir / "page_0001_asset_01.png"
        image_path.write_bytes(b"fake-image")
        annotation_path.write_text(
            json.dumps(
                {
                    "id": "asset-1",
                    "pair_type": "visual_asset",
                    "image_path": str(image_path.relative_to(ROOT)).replace("\\", "/"),
                    "asset_type": "chart",
                    "page_type": "chart",
                    "summary": "BTCUSDT H1 chart",
                    "description": "Trading chart for BTCUSDT on H1.",
                    "context_heading": "Im H1 waren wir heute Bearisch",
                    "context_text": "Im H1 waren wir heute Bearisch",
                    "ocr_text": "Bitcoin / TetherUS, 1h, BINANCE",
                    "combined_text": "Im H1 waren wir heute Bearisch\nBitcoin / TetherUS, 1h, BINANCE",
                    "primary_symbol": "BTCUSDT",
                    "instrument_name": "Bitcoin / TetherUS",
                    "venue": "BINANCE",
                    "timeframes": ["H1"],
                    "bias": "bearish",
                    "direction": None,
                    "setup_status": None,
                    "trade_levels": {},
                    "trading_concepts": [],
                    "labels": {"contains_symbol": True, "contains_timeframe": True},
                },
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

        try:
            jobs, context_by_id = collect_annotation_jobs(
                glob_pattern=str((temp_root / "annotations" / "*" / "assets" / "*.json").relative_to(ROOT)).replace("\\", "/"),
                language="en",
                skip_existing_llm=True,
                limit=None,
                max_assets_per_chat=1,
            )
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        self.assertEqual(len(jobs), 1)
        self.assertTrue(jobs[0].new_chat)
        self.assertIn("asset-1", context_by_id)

    def test_collect_annotation_jobs_rotates_chat_after_configured_batch_size(self) -> None:
        temp_root = ROOT / ".tmp" / f"test_collect_annotation_jobs_rotation_{uuid.uuid4().hex}"
        annotation_dir = temp_root / "annotations" / "03-10" / "assets"
        image_dir = temp_root / "images" / "03-10" / "assets"
        annotation_dir.mkdir(parents=True, exist_ok=True)
        image_dir.mkdir(parents=True, exist_ok=True)

        for index in range(3):
            asset_id = f"asset-{index + 1}"
            annotation_path = annotation_dir / f"page_0001_asset_{index + 1:02d}.json"
            image_path = image_dir / f"page_0001_asset_{index + 1:02d}.png"
            image_path.write_bytes(b"fake-image")
            annotation_path.write_text(
                json.dumps(
                    {
                        "id": asset_id,
                        "pair_type": "visual_asset",
                        "image_path": str(image_path.relative_to(ROOT)).replace("\\", "/"),
                        "asset_type": "chart",
                        "page_type": "chart",
                        "summary": f"BTCUSDT H1 chart {index + 1}",
                        "description": "Trading chart for BTCUSDT on H1.",
                        "context_heading": "Im H1 waren wir heute Bearisch",
                        "context_text": "Im H1 waren wir heute Bearisch",
                        "ocr_text": "Bitcoin / TetherUS, 1h, BINANCE",
                        "combined_text": "Im H1 waren wir heute Bearisch\nBitcoin / TetherUS, 1h, BINANCE",
                        "primary_symbol": "BTCUSDT",
                        "instrument_name": "Bitcoin / TetherUS",
                        "venue": "BINANCE",
                        "timeframes": ["H1"],
                        "bias": "bearish",
                        "direction": None,
                        "setup_status": None,
                        "trade_levels": {},
                        "trading_concepts": [],
                        "labels": {"contains_symbol": True, "contains_timeframe": True},
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )

        try:
            jobs, _context_by_id = collect_annotation_jobs(
                glob_pattern=str((temp_root / "annotations" / "*" / "assets" / "*.json").relative_to(ROOT)).replace("\\", "/"),
                language="en",
                skip_existing_llm=True,
                limit=None,
                max_assets_per_chat=2,
            )
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

        self.assertEqual([job.job_id for job in jobs], ["asset-1", "asset-2", "asset-3"])
        self.assertEqual([job.new_chat for job in jobs], [True, False, True])


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
            "page_type_confidence": 0.8,
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
        self.assertEqual(updated["summary"], "A trading chart with candles and a price scale.")
        self.assertEqual(updated["description"], "A BTCUSDT H1 chart with visible candles and price labels.")
        self.assertIn("chatgpt_browser_llm", updated["extraction_methods"])
        self.assertEqual(updated["target_json"]["description"]["key_visual_elements"], ["candlesticks", "price_scale"])
        self.assertIn("Short caption:", updated["clean_text"])
        self.assertEqual(
            updated["target_json"]["observed"]["visible_in_crop"]["clean_text"],
            "old clean text\nVisible labels: BTCUSDT 1h",
        )
        self.assertFalse(updated["review_required"])
        self.assertEqual(updated["target_json"]["provenance"]["quality"]["annotation_quality"], "high")
        self.assertEqual(updated["llm_enrichment"]["model_slug"], "gpt-test")

    def test_apply_asset_llm_enrichment_replaces_fragmented_visible_text_with_literal_ocr_line(self) -> None:
        annotation = {
            "caption": "old caption",
            "summary": "old summary",
            "description": "old description",
            "clean_text": "old clean text",
            "extraction_methods": ["ocr"],
            "primary_symbol": "BTCUSDT",
            "instrument_name": "Bitcoin / TetherUS",
            "timeframes": ["H1"],
            "venue": "BINANCE",
            "ocr_text": (
                "Fynn160k freigegeben fur TradingView.com, Okt 08, 2024 08:36 UTC+2\n"
                "Bitcoin / TetherUS, 1h, BINANCE\n"
                "All-In-One Sessions, Weekly, Monday, Previous Highs/Lows"
            ),
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
                "visible_text": "Bitcoin / TetherUS, 1h, BINANCE ... H1 BOS ... TradingView",
                "training_tags": ["chart", "candles"],
                "confidence": "high",
            },
            raw_response_text='{"short_caption":"BTCUSDT H1 candlestick chart"}',
            prompt="describe image",
            language="en",
            model_slug="gpt-test",
            conversation_url="https://chatgpt.com/c/test",
        )

        self.assertIn("Visible text: Bitcoin / TetherUS, 1h, BINANCE", updated["clean_text"])
        self.assertEqual(
            updated["llm_enrichment"]["structured_response"]["visible_text"],
            "Bitcoin / TetherUS, 1h, BINANCE",
        )
        self.assertEqual(
            updated["target_json"]["observed"]["visible_in_crop"]["clean_text"],
            "old clean text\nVisible labels: Bitcoin / TetherUS, 1h, BINANCE",
        )

    def test_apply_asset_llm_enrichment_backfills_market_fields_and_clears_review(self) -> None:
        annotation = {
            "caption": "old caption",
            "summary": "old summary",
            "description": "old description",
            "clean_text": "old clean text",
            "review_required": True,
            "page_type_confidence": 0.5,
            "extraction_methods": ["ocr"],
            "primary_symbol": None,
            "instrument_name": None,
            "venue": None,
            "symbols": [],
            "timeframes": ["H1"],
            "labels": {
                "contains_symbol": False,
                "contains_timeframe": True,
                "text_density": "medium",
            },
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
                        "normalized_fields": {
                            "primary_symbol": None,
                            "instrument_name": None,
                            "venue": None,
                            "symbols": [],
                            "timeframes": ["H1"],
                        },
                    }
                },
                "derived": {
                    "primary_symbol": None,
                    "instrument_name": None,
                    "venue": None,
                    "symbols": [],
                    "timeframes": ["H1"],
                },
                "provenance": {
                    "extraction_methods": ["ocr"],
                    "field_sources": {
                        "primary_symbol": "missing",
                        "instrument_name": "missing",
                        "venue": "missing",
                        "symbols": "missing",
                        "timeframes": "ocr",
                    },
                },
            },
        }

        updated = apply_asset_llm_enrichment(
            annotation=annotation,
            response_payload={
                "short_caption": "TradingView EUR/USD chart with session zones",
                "visual_summary": "A grayscale intraday EUR/USD chart with highlighted session zones and arrows.",
                "context_augmented_summary": "The image appears to show EUR/USD on a 5-minute chart from FXCM.",
                "key_visual_elements": ["candlesticks", "session_zones"],
                "limitations": ["small interface text is partially unreadable"],
                "visible_text": "Euro / US-Dollar - 5 - FXCM",
                "training_tags": ["chart", "forex", "eurusd"],
                "confidence": "high",
            },
            raw_response_text='{"short_caption":"TradingView EUR/USD chart with session zones"}',
            prompt="describe image",
            language="en",
            model_slug="gpt-test",
            conversation_url="https://chatgpt.com/c/test",
        )

        self.assertEqual(updated["primary_symbol"], "EURUSD")
        self.assertEqual(updated["instrument_name"], "Euro / US-Dollar")
        self.assertEqual(updated["venue"], "FXCM")
        self.assertEqual(updated["timeframes"], ["M5"])
        self.assertIn("EURUSD", updated["symbols"])
        self.assertFalse(updated["review_required"])
        self.assertEqual(updated["target_json"]["provenance"]["quality"]["annotation_quality"], "high")
        self.assertEqual(
            updated["target_json"]["observed"]["visible_in_crop"]["normalized_fields"]["primary_symbol"],
            "EURUSD",
        )
        self.assertEqual(
            updated["target_json"]["provenance"]["field_sources"]["primary_symbol"],
            "llm_enrichment",
        )

    def test_build_multimodal_description_prompt_forbids_joined_visible_text_fragments(self) -> None:
        prompt = build_multimodal_description_prompt(
            {
                "id": "asset-1",
                "asset_type": "chart",
                "page_type": "chart",
                "summary": "BTCUSDT H1 chart",
                "description": "Trading chart for BTCUSDT on H1.",
                "context_heading": "Im H1 waren wir heute Bearisch",
                "context_text": "Im H1 waren wir heute Bearisch",
                "ocr_text": "Bitcoin / TetherUS, 1h, BINANCE",
                "combined_text": "Im H1 waren wir heute Bearisch\nBitcoin / TetherUS, 1h, BINANCE",
                "primary_symbol": "BTCUSDT",
                "instrument_name": "Bitcoin / TetherUS",
                "venue": "BINANCE",
                "timeframes": ["H1"],
                "bias": "bearish",
                "direction": None,
                "setup_status": None,
                "trade_levels": {},
                "trading_concepts": [],
                "labels": {"contains_symbol": True, "contains_timeframe": True},
            }
        )

        self.assertIn("do not join separate fragments with ellipses", prompt)

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

    def test_apply_asset_llm_enrichment_marks_review_required_for_low_confidence_and_noisy_ocr(self) -> None:
        annotation = {
            "caption": "old caption",
            "summary": "old summary",
            "description": "old description",
            "clean_text": "old clean text",
            "page_type_confidence": 0.5,
            "ocr_text": "Â® noisy header\nBitcoin / TetherUS, 1h, BINANCE",
            "extraction_methods": ["ocr"],
            "primary_symbol": "BTCUSDT",
            "instrument_name": "Bitcoin / TetherUS",
            "timeframes": ["H1"],
            "venue": "BINANCE",
            "labels": {"text_density": "medium"},
            "target_json": {
                "description": {
                    "short_caption": "old caption",
                    "visual_summary": "old visual summary",
                    "context_augmented_summary": "old context summary",
                    "key_visual_elements": [],
                    "limitations": [],
                },
                "observed": {
                    "visible_in_crop": {
                        "clean_text": "crop text",
                    }
                },
                "provenance": {
                    "extraction_methods": ["ocr"],
                    "quality": {
                        "annotation_quality": "high",
                        "page_type_confidence": 0.5,
                    },
                    "review": {
                        "required": False,
                        "reasons": [],
                    },
                },
            },
        }

        updated = apply_asset_llm_enrichment(
            annotation=annotation,
            response_payload={
                "short_caption": "BTCUSDT H1 chart",
                "visual_summary": "A chart with candles and overlays.",
                "context_augmented_summary": "A BTCUSDT H1 chart with bearish movement.",
                "limitations": ["small text is partially unreadable"],
                "visible_text": "Bitcoin / TetherUS, 1h, BINANCE",
                "confidence": "high",
            },
            raw_response_text='{"short_caption":"BTCUSDT H1 chart"}',
            prompt="describe image",
            language="en",
            model_slug="gpt-test",
            conversation_url="https://chatgpt.com/c/test",
        )

        self.assertTrue(updated["review_required"])
        self.assertEqual(updated["target_json"]["provenance"]["quality"]["annotation_quality"], "high")
        self.assertEqual(
            updated["target_json"]["provenance"]["review"]["reasons"],
            ["ocr_encoding_artifacts"],
        )


if __name__ == "__main__":
    unittest.main()
