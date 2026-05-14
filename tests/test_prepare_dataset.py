import shutil
import sys
import unittest
import uuid
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_dataset import (
    CONCEPT_KEYWORDS,
    asset_render_scale,
    build_asset_pair_id,
    build_asset_pair_paths,
    build_asset_training_target,
    derive_asset_market_fields,
    extract_keyword_labels,
    file_sha256,
    extract_pair_symbols,
    preserve_existing_llm_enrichment,
    merge_page_market_fields,
    resolve_tesseract_cmd,
    should_review_page_annotation,
)


class PrepareDatasetTests(unittest.TestCase):
    def test_resolve_tesseract_cmd_uses_tesseract_from_path(self) -> None:
        with (
            mock.patch.dict("os.environ", {"TESSERACT_CMD": ""}, clear=False),
            mock.patch("prepare_dataset.Path.exists", return_value=False),
            mock.patch("prepare_dataset.shutil.which", return_value="/usr/bin/tesseract"),
        ):
            self.assertEqual(resolve_tesseract_cmd(), "/usr/bin/tesseract")

    def test_build_asset_pair_paths_uses_single_pairs_directory_and_shared_basename(self) -> None:
        image_path, json_path = build_asset_pair_paths("nested/03.10.pdf", page_number=1, asset_index=2, image_extension="jpeg")
        expected_basename = f"nested__03-10__p0001__a02__{build_asset_pair_id('nested/03.10.pdf', 1, 2)}"

        self.assertTrue(image_path.as_posix().endswith(f"data/processed/multimodal/pairs/{expected_basename}.jpg"))
        self.assertTrue(json_path.as_posix().endswith(f"data/processed/multimodal/pairs/{expected_basename}.json"))
        self.assertEqual(image_path.stem, json_path.stem)

    def test_preserve_existing_llm_enrichment_only_when_image_hash_matches(self) -> None:
        temp_root = ROOT / ".tmp" / f"test_preserve_existing_llm_{uuid.uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=True)
        image_path = temp_root / "asset.png"
        image_path.write_bytes(b"same-image")

        current_annotation = {
            "id": "asset-1",
            "pair_type": "visual_asset",
            "image_path": image_path.relative_to(ROOT).as_posix(),
            "caption": "new caption",
            "summary": "new summary",
            "review_required": True,
            "target_json": {
                "description": {"short_caption": "new caption"},
                "observed": {"visible_in_crop": {"clean_text": "new ocr"}},
                "derived": {"symbols": ["NEW"]},
                "provenance": {
                    "extraction_methods": ["ocr"],
                    "field_sources": {"primary_symbol": "ocr"},
                    "quality": {
                        "annotation_quality": "low",
                        "page_type_confidence": 0.1,
                    },
                    "review": {
                        "required": True,
                        "reasons": ["auto_review_flag", "missing_primary_symbol"],
                    },
                },
            },
        }
        old_annotation = {
            **current_annotation,
            "caption": "old llm caption",
            "summary": "old llm summary",
            "review_required": False,
            "target_json": {
                "description": {
                    "short_caption": "old llm caption",
                    "visual_summary": "old llm summary",
                },
                "observed": {"visible_in_crop": {"clean_text": "stale ocr"}},
                "derived": {"symbols": ["OLD"]},
                "provenance": {
                    "extraction_methods": ["ocr", "chatgpt_browser_llm"],
                    "field_sources": {
                        "primary_symbol": "llm_enrichment",
                        "venue": "ocr",
                    },
                    "quality": {
                        "annotation_quality": "high",
                        "page_type_confidence": 0.7,
                    },
                    "review": {
                        "required": False,
                        "reasons": [],
                    },
                },
            },
            "llm_enrichment": {"structured_response": {"confidence": "high"}},
        }
        existing = {
            "asset-1": {
                "annotation": old_annotation,
                "image_sha256": file_sha256(image_path),
            }
        }

        try:
            preserved, status = preserve_existing_llm_enrichment(current_annotation, existing)

            self.assertEqual(status, "preserved")
            self.assertEqual(preserved["caption"], "old llm caption")
            self.assertFalse(preserved["review_required"])
            self.assertEqual(preserved["llm_enrichment"], old_annotation["llm_enrichment"])
            self.assertEqual(preserved["target_json"]["description"]["short_caption"], "old llm caption")
            self.assertEqual(preserved["target_json"]["description"]["visual_summary"], "old llm summary")
            self.assertEqual(preserved["target_json"]["observed"]["visible_in_crop"]["clean_text"], "new ocr")
            self.assertEqual(preserved["target_json"]["derived"]["symbols"], ["NEW"])
            self.assertEqual(
                preserved["target_json"]["provenance"]["field_sources"],
                {"primary_symbol": "ocr"},
            )
            self.assertEqual(
                preserved["target_json"]["provenance"]["extraction_methods"],
                ["ocr", "chatgpt_browser_llm"],
            )
            self.assertEqual(
                preserved["target_json"]["provenance"]["quality"],
                {
                    "annotation_quality": "high",
                    "page_type_confidence": 0.7,
                },
            )
            self.assertEqual(
                preserved["target_json"]["provenance"]["review"],
                {
                    "required": False,
                    "reasons": [],
                },
            )

            image_path.write_bytes(b"changed-image")
            unchanged, status = preserve_existing_llm_enrichment(current_annotation, existing)

            self.assertEqual(status, "image_changed")
            self.assertNotIn("llm_enrichment", unchanged)
            self.assertEqual(unchanged["caption"], "new caption")
            self.assertTrue(unchanged["review_required"])
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_preserve_existing_llm_enrichment_uses_unique_hash_when_asset_id_shifts(self) -> None:
        temp_root = ROOT / ".tmp" / f"test_preserve_existing_shifted_llm_{uuid.uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=True)
        image_path = temp_root / "asset.png"
        image_path.write_bytes(b"same-image")

        current_annotation = {
            "id": "asset-2",
            "pair_type": "visual_asset",
            "image_path": image_path.relative_to(ROOT).as_posix(),
            "caption": "new caption",
            "review_required": True,
            "target_json": {
                "description": {"short_caption": "new caption"},
                "provenance": {"extraction_methods": ["ocr"]},
            },
        }
        shifted_old_annotation = {
            **current_annotation,
            "id": "asset-1",
            "caption": "old shifted llm caption",
            "review_required": False,
            "target_json": {
                "description": {
                    "short_caption": "old shifted llm caption",
                    "visual_summary": "old shifted summary",
                },
                "provenance": {"extraction_methods": ["ocr", "chatgpt_browser_llm"]},
            },
            "llm_enrichment": {"structured_response": {"confidence": "high"}},
        }
        old_annotation_with_colliding_current_id = {
            **current_annotation,
            "caption": "different old asset",
            "llm_enrichment": {"structured_response": {"confidence": "low"}},
        }
        existing = {
            "asset-1": {
                "annotation": shifted_old_annotation,
                "image_sha256": file_sha256(image_path),
            },
            "asset-2": {
                "annotation": old_annotation_with_colliding_current_id,
                "image_sha256": "different-image-hash",
            },
        }

        try:
            preserved, status = preserve_existing_llm_enrichment(current_annotation, existing)

            self.assertEqual(status, "preserved")
            self.assertEqual(preserved["caption"], "old shifted llm caption")
            self.assertFalse(preserved["review_required"])
            self.assertEqual(preserved["llm_enrichment"], shifted_old_annotation["llm_enrichment"])
            self.assertEqual(
                preserved["target_json"]["description"]["visual_summary"],
                "old shifted summary",
            )
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_preserve_existing_llm_enrichment_skips_ambiguous_duplicate_hash(self) -> None:
        temp_root = ROOT / ".tmp" / f"test_preserve_existing_duplicate_llm_{uuid.uuid4().hex}"
        temp_root.mkdir(parents=True, exist_ok=True)
        image_path = temp_root / "asset.png"
        image_path.write_bytes(b"same-image")

        current_annotation = {
            "id": "asset-3",
            "pair_type": "visual_asset",
            "image_path": image_path.relative_to(ROOT).as_posix(),
            "caption": "new caption",
            "review_required": True,
        }
        existing = {
            "asset-1": {
                "annotation": {
                    **current_annotation,
                    "id": "asset-1",
                    "caption": "first old caption",
                    "llm_enrichment": {"structured_response": {"confidence": "high"}},
                },
                "image_sha256": file_sha256(image_path),
            },
            "asset-2": {
                "annotation": {
                    **current_annotation,
                    "id": "asset-2",
                    "caption": "second old caption",
                    "llm_enrichment": {"structured_response": {"confidence": "medium"}},
                },
                "image_sha256": file_sha256(image_path),
            },
        }

        try:
            unchanged, status = preserve_existing_llm_enrichment(current_annotation, existing)

            self.assertEqual(status, "duplicate_image_hash")
            self.assertNotIn("llm_enrichment", unchanged)
            self.assertEqual(unchanged["caption"], "new caption")
            self.assertTrue(unchanged["review_required"])
        finally:
            shutil.rmtree(temp_root, ignore_errors=True)

    def test_asset_render_scale_prefers_higher_quality_but_caps_maximum(self) -> None:
        self.assertEqual(asset_render_scale((0.0, 0.0, 500.0, 250.0)), 4.0)
        self.assertEqual(asset_render_scale((0.0, 0.0, 1400.0, 700.0)), 2.0)

    def test_extract_keyword_labels_ignores_rr_ocr_artifact(self) -> None:
        text = "59.500,00\n59.2000\nRR --------\n0 12:00 1:00 2 05:00"

        self.assertNotIn("risk_management", extract_keyword_labels(text, CONCEPT_KEYWORDS))

    def test_extract_pair_symbols_ignores_multiline_noise_before_visible_pair(self) -> None:
        text = (
            "Im M15 gab es heute kein Setup\n"
            "i | oF | FY SD\n"
            "l i LY\n"
            "Bitcoin / TetherUS, 15, BINANCE\n"
        )

        self.assertEqual(extract_pair_symbols(text), ["BTCUSDT"])

    def test_derive_asset_market_fields_prefers_visible_symbol_over_context_noise(self) -> None:
        fields = derive_asset_market_fields(
            context_text="Im M15 gab es heute kein Setup. NAS100USDT wurde spaeter auf der Seite erwaehnt.",
            ocr_text="Bitcoin / TetherUS, 15, BINANCE",
            fallback_text=(
                "Im M15 gab es heute kein Setup. NAS100USDT wurde spaeter auf der Seite erwaehnt.\n"
                "Bitcoin / TetherUS, 15, BINANCE"
            ),
        )

        self.assertEqual(fields["symbols"], ["BTCUSDT"])
        self.assertEqual(fields["primary_symbol"], "BTCUSDT")
        self.assertEqual(fields["instrument_name"], "Bitcoin / TetherUS")
        self.assertEqual(fields["venue"], "BINANCE")
        self.assertEqual(fields["timeframes"], ["M15"])

    def test_build_asset_training_target_filters_visible_fields_per_value(self) -> None:
        annotation = {
            "caption": "BTCUSDT M15 chart",
            "description": "Trading chart for BTCUSDT on M15. NAS100USDT appears in nearby page text only.",
            "context_heading": "Im M15 gab es heute kein Setup",
            "context_text": "Im M15 gab es heute kein Setup. NAS100USDT wurde spaeter auf der Seite erwaehnt.",
            "ocr_text": "Bitcoin / TetherUS, 15, BINANCE",
            "primary_symbol": "BTCUSDT",
            "instrument_name": "Bitcoin / TetherUS",
            "venue": "BINANCE",
            "symbols": ["BTCUSDT", "NAS100USDT"],
            "timeframes": ["M15", "H1"],
            "bias": None,
            "direction": None,
            "setup_status": "no_setup",
            "trade_levels": {},
            "trading_concepts": [],
            "trading_domains": [],
            "asset_type": "chart",
            "page_type": "chart",
            "labels": {
                "contains_symbol": True,
                "contains_timeframe": True,
                "contains_trade_levels": False,
                "contains_performance_metrics": False,
                "contains_strategy_rules": False,
                "likely_chart": True,
                "text_density": "medium",
                "has_context_text": True,
                "has_ocr_text": True,
                "is_large_visual": True,
                "paired_text_blocks": 1,
            },
            "source_pdf": "03.10.pdf",
            "page_number": 1,
            "asset_index": 2,
            "asset_source": "embedded_image",
            "extraction_methods": ["nearby_text", "ocr"],
            "page_type_confidence": 0.5,
            "review_required": False,
            "ocr_metadata": {
                "enabled": True,
                "available": True,
                "status": "ok",
                "language": "eng",
            },
        }

        target = build_asset_training_target(annotation)

        self.assertEqual(
            target["observed"]["visible_in_crop"]["normalized_fields"]["symbols"],
            ["BTCUSDT"],
        )
        self.assertEqual(
            target["observed"]["visible_in_crop"]["normalized_fields"]["timeframes"],
            ["M15"],
        )
        self.assertEqual(target["derived"]["symbols"], ["BTCUSDT", "NAS100USDT"])

    def test_build_asset_training_target_does_not_promote_context_only_symbol_to_visible_crop(self) -> None:
        annotation = {
            "caption": "NAS100USDT chart",
            "description": "NAS100USDT is only mentioned in nearby context.",
            "context_heading": "Session note",
            "context_text": "Nearby page text mentions NAS100USDT and an H1 setup.",
            "ocr_text": "Volume profile, H1, BINANCE",
            "primary_symbol": "NAS100USDT",
            "instrument_name": "Nasdaq 100 / TetherUS",
            "venue": "BINANCE",
            "symbols": ["NAS100USDT"],
            "timeframes": ["H1"],
            "bias": None,
            "direction": None,
            "setup_status": None,
            "trade_levels": {},
            "trading_concepts": [],
            "trading_domains": [],
            "asset_type": "chart",
            "page_type": "chart",
            "labels": {
                "contains_symbol": True,
                "contains_timeframe": True,
                "contains_trade_levels": False,
                "contains_performance_metrics": False,
                "contains_strategy_rules": False,
                "likely_chart": True,
                "text_density": "medium",
                "has_context_text": True,
                "has_ocr_text": True,
                "is_large_visual": True,
                "paired_text_blocks": 1,
            },
            "source_pdf": "03.10.pdf",
            "page_number": 1,
            "asset_index": 3,
            "asset_source": "embedded_image",
            "extraction_methods": ["nearby_text", "ocr"],
            "page_type_confidence": 0.5,
            "review_required": False,
            "ocr_metadata": {
                "enabled": True,
                "available": True,
                "status": "ok",
                "language": "eng",
            },
        }

        target = build_asset_training_target(annotation)

        self.assertIsNone(target["observed"]["visible_in_crop"]["normalized_fields"]["primary_symbol"])
        self.assertIsNone(target["observed"]["visible_in_crop"]["normalized_fields"]["instrument_name"])
        self.assertEqual(target["observed"]["visible_in_crop"]["normalized_fields"]["timeframes"], ["H1"])
        self.assertEqual(target["provenance"]["field_sources"]["primary_symbol"], "context")

    def test_build_asset_training_target_keeps_ocr_normalized_venue_visible(self) -> None:
        annotation = {
            "caption": "BTCUSDT M15 chart",
            "description": "Trading chart for BTCUSDT on M15.",
            "context_heading": "Context",
            "context_text": "BTCUSDT on M15 with venue mentioned elsewhere.",
            "ocr_text": "Bitcoin / TetherUS, 15, BIMANCE",
            "primary_symbol": "BTCUSDT",
            "instrument_name": "Bitcoin / TetherUS",
            "venue": "BINANCE",
            "symbols": ["BTCUSDT"],
            "timeframes": ["M15"],
            "bias": None,
            "direction": None,
            "setup_status": None,
            "trade_levels": {},
            "trading_concepts": [],
            "trading_domains": [],
            "asset_type": "chart",
            "page_type": "chart",
            "labels": {
                "contains_symbol": True,
                "contains_timeframe": True,
                "contains_trade_levels": False,
                "contains_performance_metrics": False,
                "contains_strategy_rules": False,
                "likely_chart": True,
                "text_density": "medium",
                "has_context_text": True,
                "has_ocr_text": True,
                "is_large_visual": True,
                "paired_text_blocks": 1,
            },
            "source_pdf": "03.10.pdf",
            "page_number": 1,
            "asset_index": 4,
            "asset_source": "embedded_image",
            "extraction_methods": ["nearby_text", "ocr"],
            "page_type_confidence": 0.5,
            "review_required": False,
            "ocr_metadata": {
                "enabled": True,
                "available": True,
                "status": "ok",
                "language": "eng",
            },
        }

        target = build_asset_training_target(annotation)

        self.assertEqual(target["observed"]["visible_in_crop"]["normalized_fields"]["venue"], "BINANCE")
        self.assertEqual(target["provenance"]["field_sources"]["venue"], "ocr_normalization")

    def test_build_asset_training_target_marks_fuzzy_ocr_symbol_as_visible(self) -> None:
        annotation = {
            "caption": "BTCUSDT H1 chart",
            "description": "Trading chart for BTCUSDT on H1.",
            "context_heading": "Context",
            "context_text": "Bearish H1 setup.",
            "ocr_text": "Bitcon / TetherUS, 1h, BINANCE",
            "primary_symbol": "BTCUSDT",
            "instrument_name": "Bitcoin / TetherUS",
            "venue": "BINANCE",
            "symbols": ["BTCUSDT"],
            "timeframes": ["H1"],
            "bias": None,
            "direction": None,
            "setup_status": None,
            "trade_levels": {},
            "trading_concepts": [],
            "trading_domains": [],
            "asset_type": "chart",
            "page_type": "chart",
            "labels": {
                "contains_symbol": True,
                "contains_timeframe": True,
                "contains_trade_levels": False,
                "contains_performance_metrics": False,
                "contains_strategy_rules": False,
                "likely_chart": True,
                "text_density": "medium",
                "has_context_text": True,
                "has_ocr_text": True,
                "is_large_visual": True,
                "paired_text_blocks": 1,
            },
            "source_pdf": "03.10.pdf",
            "page_number": 1,
            "asset_index": 5,
            "asset_source": "embedded_image",
            "extraction_methods": ["nearby_text", "ocr"],
            "page_type_confidence": 0.5,
            "review_required": False,
            "ocr_metadata": {
                "enabled": True,
                "available": True,
                "status": "ok",
                "language": "eng",
            },
        }

        target = build_asset_training_target(annotation)

        self.assertEqual(target["observed"]["visible_in_crop"]["normalized_fields"]["primary_symbol"], "BTCUSDT")
        self.assertEqual(target["observed"]["visible_in_crop"]["normalized_fields"]["symbols"], ["BTCUSDT"])
        self.assertEqual(target["provenance"]["field_sources"]["primary_symbol"], "ocr_normalization")

    def test_merge_page_market_fields_uses_asset_symbols_when_page_ocr_misses_symbol(self) -> None:
        symbols, timeframes = merge_page_market_fields(
            symbols=[],
            timeframes=["H1"],
            asset_annotations=[
                {"symbols": ["BTCUSDT"], "timeframes": ["H1"]},
                {"symbols": ["BTCUSDT"], "timeframes": ["M15"]},
            ],
        )

        self.assertEqual(symbols, ["BTCUSDT"])
        self.assertEqual(timeframes, ["H1", "M15"])

    def test_should_review_page_annotation_when_assets_exist_but_symbol_is_missing(self) -> None:
        self.assertTrue(
            should_review_page_annotation(
                combined_text="Im H1 waren wir heute Bearisch\nIm M15 gab es heute kein Setup",
                labels={"text_density": "medium"},
                page_type="chart",
                page_type_confidence=0.5,
                ocr_text="03.10.2024\nRR --------",
                symbols=[],
                timeframes=["H1", "M15"],
                asset_count=2,
            )
        )


if __name__ == "__main__":
    unittest.main()
