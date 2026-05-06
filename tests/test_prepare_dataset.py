import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_dataset import (
    CONCEPT_KEYWORDS,
    build_asset_training_target,
    derive_asset_market_fields,
    extract_keyword_labels,
    extract_pair_symbols,
    merge_page_market_fields,
    should_review_page_annotation,
)


class PrepareDatasetTests(unittest.TestCase):
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
